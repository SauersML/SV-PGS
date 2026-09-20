"""Fewer store reads for Stage 2's EP oracle, every certified answer unchanged (lane speed-recycle).

A mean solve (``DualGaussian.iterate``) pays reads that are not Krylov iterations: X m for the right-hand side, an exact
residual at the warm start, another to certify the end, two to finish (X~'z for the mean, then X mu), and one more for
the resolved sites' columns. Their algebra lets most of them share reads:

1. **A fused start.** The warm start's exact residual is b - S z0 = (I - H) W^1/2 (y~ - X mu0) - z0, with
   mu0 = m + D X~'z0 the mean the start implies. So the read that forms X m also forms S z0: per block, X_b'T0 and then
   one X_b product of [D_b X_b'T0, m_b]. The resolved sites' genotype columns ride the same read, and are kept (raw)
   while a model's resolved set does not change.
2. **A fused finish.** The read that certifies a solve (its exact residual, S z) also finishes it: the same X_b'T gives
   the mean mu = m + D X~'z block by block, and X mu is algebra on what that read produced (X m from the start,
   X D X~'z from the residual's own product, X_L mu_L from the kept columns). ``certified_block_cg(confirm=False)``
   stops before its own certifying read, and the fused finish reads once in its place. When the start already meets
   every bound, the start read is also the finish.
3. **Posterior and information solves** (the move check, B's linear response, the cavity certificate) end the same way:
   the read that certifies their duals also forms X~'z, from which the solution follows by the split's algebra, with
   the resolved columns' products kept from the mean solve. A tightened bound on the same right-hand side continues
   from the last duals and their exact residual (no new image of the right-hand side, no restart read).
4. **Restores cost no read.** A fixed point keeps the solver's state with its snapshot, so ``ensure()`` after a later
   trial moved the solver puts it back instead of solving the mean again.
5. **Frozen EP passes without probes, at the accuracy they need.** With the cavity precisions frozen, the probe columns
   (read only by the next refresh) are dead weight, and a pass whose only use is the next site update needs its mean
   only to the forcing accuracy ``optimal_forcing`` derives. The exit is decided only on a pass solved to the
   certificate's own bound, so the exit rule, and everything after it, is ``full_data_fit``'s.

Everything is exact algebra or a certified bound; no number is set by hand. ``fit_full_data`` is
``full_data_fit.fit_full_data`` with these, and returns the same ``FullDataFit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from sv_pgs._typing import F64Array
from sv_pgs.dual_solve import (
    Deflation,
    DualCertificate,
    DualGaussian,
    DualModels,
    _cholesky_solve,
    _host,
    certified_block_cg,
    mean_right_hand_side,
    resolved_block,
    resolved_spikes,
    split_columns,
)
from sv_pgs.full_data_fit import FitCertificate, FullDataFit, NoFixedPoint, _FullDataFixedPoints, _norm_bounds, moment_starts
from sv_pgs.krylov_recycle import local_response
from sv_pgs.marginal_variances import BlockGrams, BulkSolve, WindowCross, variance_jvp
from sv_pgs.scale_mixture_ep import Cavity, FixedPoint, GaussianPosterior, MixtureHyperparameters, ScaleMixturePrior, fit_hyperparameters, initial_hyperparameters

_EPSILON = float(np.finfo(np.float64).eps)
# The solver's attributes one mean solve sets: what a snapshot keeps so a restore needs no read (item 4).
_SOLVER_STATE = (
    "mean", "genetic_image", "alpha", "linear_predictor", "noise_variance", "bulk_solves", "_duals", "_resolved", "_state", "_layout",
    "_probe_duals", "_probe_resolved", "_products", "_prior_image",
)


def optimal_forcing(contraction: float, rate: float, fixed_reads: float) -> float:
    """The relative solve accuracy eta per EP pass that minimizes the reads to converge (item 5 of the docstring).

    Mean-only EP with frozen cavities contracts its error by ``contraction`` rho per exact pass. A pass solved to a
    relative error eta (of the last move) contracts by at most rho + eta, so reaching a fixed reduction takes
    N(eta) = const / ln(1 / (rho + eta)) passes. A warm-started pass reduces its residual from about (rho + eta) to eta
    of that scale, ln(1 + rho / eta) / c Krylov reads at the solver's per-read contraction c = ``rate``, plus
    ``fixed_reads`` reads that do not depend on eta. So

        T(eta) = (fixed_reads + ln(1 + rho / eta) / c) / ln(1 / (rho + eta)),   0 < eta < 1 - rho,

    which tends to infinity at both ends; bisection on the sign of its derivative finds the interior minimum to float64
    resolution. Returns 0 (solve to the certificate's own bound) when there is no room: rho at or past 1, or no rate.
    """
    if not contraction < 1.0 or not rate > 0.0:
        return 0.0
    contraction = max(contraction, 0.0)

    def slope(eta: float) -> float:
        numerator = fixed_reads + np.log1p(contraction / eta) / rate
        numerator_slope = -(contraction / (eta * (eta + contraction))) / rate
        denominator = -np.log(contraction + eta)
        return numerator_slope / numerator + 1.0 / ((contraction + eta) * denominator)

    # The largest eta whose rho + eta is below 1 in float64, so the denominator is positive at both ends.
    low, high = np.finfo(np.float64).tiny, 1.0 - contraction
    while contraction + high >= 1.0:
        high = float(np.nextafter(high, 0.0))
    if slope(high) <= 0.0:
        return high
    while high - low > _EPSILON * high:
        middle = 0.5 * (low + high)
        if slope(middle) < 0.0:
            low = middle
        else:
            high = middle
    return float(high)


@dataclass
class _Read:
    """What one fused read produced at duals z: X (D o X~'z) per column (``image``), X~'z for the kept columns
    (``products``), and z itself."""

    duals: Any
    image: Any
    products: Any


@dataclass
class PosteriorState:
    """A posterior solve's continuation: its right-hand side (the bulk image) and its duals with their exact residual
    and last fused read, valid while the solver stays at the same sites (``operator``)."""

    values: Any
    bulk_values: Any
    column_models: Any
    right_hand_side: Any
    duals: Any
    residual: Any
    read: _Read
    operator: int


class EconomicalDualGaussian(DualGaussian):
    """``DualGaussian`` with items 1-4 of the module docstring; the same certified answers in fewer reads."""

    def __init__(self, **keywords: Any) -> None:
        super().__init__(**keywords)
        self.operator = 0
        self.measured_rate = 0.0
        self._layout: dict | None = None
        self._probe_duals: Any = None
        self._probe_resolved: dict = {}
        self._raw: dict[int, tuple[np.ndarray, Any]] = {}
        self._products: dict[int, Any] = {}
        self._prior_image: Any = None

    # -- state for restores (item 4) ------------------------------------------------------------------------------
    def save(self) -> dict:
        # Before the first mean solve some of these are not set yet (the oracle snapshots at its first call).
        state = {name: getattr(self, name, None) for name in _SOLVER_STATE}
        state["_state"] = dict(self._state)
        if "blocks" in state["_state"]:
            state["_state"]["blocks"] = dict(state["_state"]["blocks"])
        # A refinement replaces a model's block and its kept products together: both are copied, so a restore never
        # pairs one snapshot's Z_L with another's X~'Z_L.
        state["_products"] = dict(self._products)
        return state

    def load(self, state: dict) -> None:
        for name in _SOLVER_STATE:
            setattr(self, name, state[name])
        self._state = dict(state["_state"])
        if "blocks" in self._state:
            self._state["blocks"] = dict(self._state["blocks"])
        self._products = dict(state["_products"])
        self.operator += 1

    # -- the fused read (items 1-3) --------------------------------------------------------------------------------
    def _read(
        self, models: DualModels, *, values: Any = None, column_models: np.ndarray | None = None, keep: np.ndarray | None = None,
        prior: Any = None, fetch: np.ndarray | None = None, label: str,
    ) -> tuple[Any, Any, Any, Any]:
        """One read of the store: X (D o X~'V) for the columns of ``values`` (n x c), X~'V on the ``keep`` columns,
        X ``prior`` (p x M), and the raw genotype columns of the variants ``fetch`` (ascending)."""
        array_module = self.array_module
        source = self.source
        width = 0 if values is None else int(values.shape[1])
        image = array_module.zeros((source.sample_count, width))
        prior_image = None if prior is None else array_module.zeros((source.sample_count, int(prior.shape[1])))
        kept = None if keep is None or not width else array_module.zeros((source.variant_count, int(keep.size)))
        device_models = None if column_models is None else array_module.asarray(column_models)
        left = None if not width else models.sample_to_design(values, device_models)
        keep_device = None if keep is None else array_module.asarray(keep)
        fetched = []
        for start, stop, tile in source.blocks():
            parts = []
            if width:
                products = tile.rmatmat(left)
                if kept is not None:
                    kept[start:stop] = products[:, keep_device]
                parts.append(models.variances[start:stop][:, device_models] * products)
            if prior is not None:
                parts.append(prior[start:stop])
            if parts:
                out = tile.matmat(array_module.ascontiguousarray(array_module.concatenate(parts, axis=1)))
                if width:
                    image += out[:, :width]
                if prior_image is not None:
                    prior_image += out[:, width:]
            if fetch is not None and fetch.size:
                inside = fetch[(fetch >= start) & (fetch < stop)]
                if inside.size:
                    fetched.append(tile.columns(inside - start))
        self.count.note(width + (0 if prior is None else int(prior.shape[1])), 0.0, label)
        columns = array_module.concatenate(fetched, axis=1) if fetched else None
        return image, prior_image, kept, columns

    def _operator_image(self, models: DualModels, values: Any, image: Any, column_models: np.ndarray) -> Any:
        """S V from a fused read's X (D o X~'V)."""
        return values + models.design_to_sample(image, self.array_module.asarray(column_models))

    def _carried_start(self, width: int, layout: dict, resolved: dict) -> Any:
        """The previous duals where the bulk operator's columns mean the same thing: an unchanged resolved set. The
        mean and resolved-design columns carry from the last solve, the probes from the last solve that had them."""
        array_module = self.array_module
        previous = self._layout
        if previous is None or self._duals is None or previous["samples"] != self.source.sample_count:
            return None
        if any(not np.array_equal(previous["resolved"][model], resolved[model]) for model in range(self.model_count)):
            return None
        start = array_module.zeros((self.source.sample_count, width))
        start[:, : self.model_count] = self._duals[:, : self.model_count]
        for model, (first, indices) in layout["designs"].items():
            old_first, _old = previous["designs"][model]
            start[:, first : first + indices.size] = self._duals[:, old_first : old_first + indices.size]
        if layout["probes"] is not None and self._probe_duals is not None and all(
            np.array_equal(self._probe_resolved[model], resolved[model]) for model in range(self.model_count)
        ):
            first, probe_width = layout["probes"]
            start[:, first : first + probe_width] = self._probe_duals
        return start

    # -- the mean solve (items 1, 2 and 5's probe drop) -----------------------------------------------------------
    def iterate(
        self, *, site_precision: Any, site_shift: Any, noise_variance: np.ndarray, error_bound: Any, probe_residual_ratio: float, with_probes: bool = True
    ) -> DualCertificate:
        """``DualGaussian.iterate`` with fused start and finish reads; with ``with_probes`` False, no probe columns
        (``bulk_solves`` then keeps the last refresh's)."""
        array_module = self.array_module
        source = self.source
        model_count = self.model_count
        self.operator += 1
        precision = array_module.asarray(site_precision, dtype=array_module.float64)
        shift = array_module.asarray(site_shift, dtype=array_module.float64)
        self.noise_variance = np.asarray(noise_variance, dtype=np.float64).copy()
        positive = precision > 0.0
        variances = array_module.where(positive, 1.0 / array_module.where(positive, precision, 1.0), 0.0)
        spikes = variances * self.unit_squares / array_module.asarray(self.noise_variance)[None, :]
        resolved: dict[int, np.ndarray] = {}
        for model in range(model_count):
            nonpositive = _host(array_module.flatnonzero(~positive[:, model]))
            candidate = array_module.where(positive[:, model], spikes[:, model], 0.0)
            resolved[model] = np.union1d(nonpositive, resolved_spikes(candidate, int(self.training_counts[model]), array_module)).astype(np.int64)
        bulk_variances = variances.copy()
        for model, indices in resolved.items():
            bulk_variances[array_module.asarray(indices), model] = 0.0
        bulk_mean = bulk_variances * shift
        models = self._models(self.noise_variance, bulk_variances)
        order = [model for model in range(model_count) if resolved[model].size]
        widths = [model_count] + [int(resolved[model].size) for model in order] + ([int(self.probes.shape[1])] if with_probes else [])
        offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
        layout = {
            "samples": source.sample_count,
            "resolved": {model: indices.copy() for model, indices in resolved.items()},
            "designs": {model: (int(offsets[position + 1]), resolved[model]) for position, model in enumerate(order)},
            "probes": (int(offsets[-2]), int(self.probes.shape[1])) if with_probes else None,
        }
        column_models = np.concatenate(
            [np.arange(model_count)] + [np.full(int(resolved[model].size), model) for model in order] + ([self.probe_models] if with_probes else [])
        )
        # The mean and resolved-design columns, whose products the finish keeps.
        keep = np.arange(int(offsets[len(order) + 1]))
        needed = np.unique(np.concatenate([resolved[model] for model in order])) if order else np.zeros(0, dtype=np.int64)
        cached = {int(index) for model in order for index in (self._raw[model][0] if model in self._raw else ())}
        fetch = np.asarray(sorted(int(index) for index in needed if int(index) not in cached), dtype=np.int64)
        start = self._carried_start(int(offsets[-1]), layout, resolved)
        if start is not None:
            image, prior_image, kept, columns = self._read(models, values=start, column_models=column_models, keep=keep, prior=bulk_mean, fetch=fetch, label="start")
        else:
            _image, prior_image, _kept, columns = self._read(models, prior=bulk_mean, fetch=fetch, label="prior-image")
        fetched = {int(index): position for position, index in enumerate(fetch)}
        raw: dict[int, tuple[np.ndarray, Any]] = {}
        for model in order:
            old_indices, old_columns = self._raw.get(model, (np.zeros(0, dtype=np.int64), None))
            old_position = {int(index): position for position, index in enumerate(old_indices)}
            pieces = []
            for index in resolved[model]:
                index = int(index)
                if index in old_position:
                    pieces.append(old_columns[:, old_position[index] : old_position[index] + 1])
                elif index in fetched:
                    pieces.append(columns[:, fetched[index] : fetched[index] + 1])
                else:
                    # Kept by another model: take it from that model's columns.
                    other = next(other for other in order if other in self._raw and index in set(self._raw[other][0].tolist()))
                    other_position = int(np.flatnonzero(self._raw[other][0] == index)[0])
                    pieces.append(self._raw[other][1][:, other_position : other_position + 1])
            raw[model] = (resolved[model].copy(), array_module.concatenate(pieces, axis=1))
        self._raw = raw
        designs = {model: models.design_to_sample(raw[model][1], array_module.full(int(resolved[model].size), model)) for model in order}
        right = mean_right_hand_side(models, self.targets - self.offsets, prior_image)
        stacked = array_module.concatenate([right] + [designs[model] for model in order] + ([self.probes] if with_probes else []), axis=1)
        target = array_module.asarray(error_bound, dtype=array_module.float64)
        column_norms = array_module.linalg.norm(stacked, axis=0)
        bound = probe_residual_ratio * column_norms
        bound[:model_count] = target
        for position, model in enumerate(order):
            relative = min(probe_residual_ratio, float(target[model]) / max(float(column_norms[model]), np.finfo(np.float64).tiny))
            bound[offsets[position + 1] : offsets[position + 2]] = relative * column_norms[offsets[position + 1] : offsets[position + 2]]
        if start is not None:
            solution = start
            residual = stacked - self._operator_image(models, start, image, column_models)
            read = _Read(start, image, kept)
        else:
            # Zero duals: their products and images are zero, so the finish needs no read of its own.
            solution = array_module.zeros_like(stacked)
            residual = stacked.copy()
            read = _Read(solution, array_module.zeros_like(stacked), array_module.zeros((source.variant_count, int(keep.size))))
        resolved_sites = {
            model: (precision[array_module.asarray(resolved[model]), model], shift[array_module.asarray(resolved[model]), model]) for model in order
        }
        spike_free = Deflation({}, {}, {}, resolved)
        device_models = array_module.asarray(column_models)
        iterations = restarts = 0
        cold = start is None
        while True:
            if bool(array_module.any(array_module.linalg.norm(residual, axis=0) > bound)):
                correction = certified_block_cg(
                    source, models, residual, array_module.zeros_like(residual), device_models, bound, self.count, deflation=spike_free, label="gaussian", confirm=False,
                )
                iterations += correction.iterations
                restarts += correction.restarts
                solution = solution + correction.solution
                image, _prior, kept, _columns = self._read(models, values=solution, column_models=column_models, keep=keep, label="finish")
                residual = stacked - self._operator_image(models, solution, image, column_models)
                read = _Read(solution, image, kept)
            mean_duals = solution[:, :model_count].copy()
            certificate = array_module.linalg.norm(residual[:, :model_count], axis=0)
            blocks_by_model: dict = {}
            resolved_means: dict[int, Any] = {}
            for position, model in enumerate(order):
                columns = slice(int(offsets[position + 1]), int(offsets[position + 2]))
                block = resolved_block(array_module, designs[model], resolved_sites[model][0], solution[:, columns], residual[:, columns])
                resolved_mean, model_duals, model_certificate = split_columns(
                    array_module, block, resolved_sites[model][1][:, None], solution[:, model : model + 1], residual[:, model : model + 1]
                )
                mean_duals[:, model] = model_duals[:, 0]
                certificate[model] = model_certificate[0]
                blocks_by_model[model] = block
                resolved_means[model] = resolved_mean[:, 0]
            open_models = _host(certificate > target)
            if not open_models.any():
                break
            for model in np.flatnonzero(open_models):
                shortfall = float(target[model] / certificate[model]) if bool(array_module.isfinite(certificate[model])) else float(target[model] / max(float(column_norms[model]), np.finfo(np.float64).tiny))
                bound[model] *= shortfall
                if model in order:
                    position = order.index(model)
                    bound[offsets[position + 1] : offsets[position + 2]] *= shortfall
        if cold and iterations:
            reduction = float(np.max(_host(column_norms[:model_count] / target)))
            if reduction > 1.0:
                # The measured per-read contraction of a cold mean solve, for the frozen passes' forcing (item 5).
                self.measured_rate = float(np.log(reduction)) / iterations
        state: dict = {"designs": designs, "order": order, "offsets": offsets, "blocks": blocks_by_model, "resolved_mean": resolved_means}
        self._duals = solution
        if with_probes:
            self._probe_duals = solution[:, layout["probes"][0] :].copy()
            self._probe_resolved = {model: indices.copy() for model, indices in resolved.items()}
        self._resolved = resolved
        self._layout = layout
        self._prior_image = prior_image
        self._state = state | {"models": models, "bulk_mean": bulk_mean, "bulk_variances": bulk_variances, "precision": precision}
        self._finish_from(read, prior_image, bulk_mean, bulk_variances, precision, models, state, resolved, with_probes, offsets)
        return DualCertificate(certificate, target, np.array([resolved[model].size for model in range(model_count)]), iterations, restarts)

    def _finish_from(
        self, read: _Read, prior_image: Any, bulk_mean: Any, bulk_variances: Any, precision: Any, models: DualModels, state: dict, resolved: dict,
        with_probes: bool, offsets: np.ndarray,
    ) -> None:
        """``DualGaussian._finish`` with no read: mu = m + D X~'z_mean (z_mean = z_b - Z_L mu_L, the split's), X mu,
        C = Xt'Z_L on the windows, and the probes' traces, all from the certifying read (item 2)."""
        array_module = self.array_module
        model_count = self.model_count
        order = state["order"]
        mean = bulk_mean.copy()
        genetic = prior_image + read.image[:, :model_count]
        self._products = {}
        for position, model in enumerate(order):
            columns = slice(int(offsets[position + 1]), int(offsets[position + 2]))
            resolved_mean = state["resolved_mean"][model]
            design_products = read.products[:, columns]
            self._products[model] = design_products
            mean[:, model] += bulk_variances[:, model] * (read.products[:, model] - design_products @ resolved_mean)
            mean[array_module.asarray(resolved[model]), model] = resolved_mean
            genetic[:, model] += self._raw[model][1] @ resolved_mean - read.image[:, columns] @ resolved_mean
        for model in range(model_count):
            if model not in state["blocks"]:
                mean[:, model] += bulk_variances[:, model] * read.products[:, model]
        column_offsets = {model: int(offsets[position + 1]) for position, model in enumerate(order)}
        positions = {model: self.windows.positions(resolved[model]) for model in order}
        window_values: dict = {model: [None] * self.windows.block_count for model in order}
        if order:
            for start, stop in self.source.block_bounds:
                self.windows.gather(array_module, read.products[start:stop], start, column_offsets, positions, window_values)
        self.mean = mean
        self.genetic_image = genetic
        weights = models.weights
        remainder = self.targets - self.offsets - genetic
        normal = array_module.einsum("na,nm,nb->mab", self.covariates, weights, self.covariates)
        self.alpha = array_module.linalg.solve(normal, (self.covariates.T @ (weights * remainder)).T[:, :, None])[:, :, 0].T
        self.linear_predictor = self.offsets + genetic + self.covariates @ self.alpha
        if not with_probes:
            return
        probe_solutions = read.duals[:, int(offsets[-2]) :]
        self.bulk_solves = []
        for model in range(model_count):
            probe_columns = array_module.asarray(np.flatnonzero(self.probe_models == model))
            probes = self.probes[:, probe_columns]
            solved = probe_solutions[:, probe_columns]
            count = float(self.training_counts[model])
            kernel = solved
            core = array_module.zeros((0, 0))
            model_cross = self.windows.empty()
            if model in state["blocks"]:
                block = state["blocks"][model]
                kernel = solved - block.duals @ _cholesky_solve(array_module, block.factor, block.design.T @ solved)
                core = block.core
                model_cross = WindowCross(positions=tuple(positions[model]), values=tuple(window_values[model]))
            self.bulk_solves.append(BulkSolve(
                site_precision=_host(precision[:, model]),
                resolved=resolved[model],
                resolved_core=_host(core),
                resolved_cross=model_cross,
                bulk_trace=float(array_module.sum(probes * solved)) / (self.probe_count * count),
                bulk_square_trace=float(array_module.sum(solved * solved)) / (self.probe_count * count),
                kernel_square_trace=float(array_module.sum(kernel * kernel)) / (self.probe_count * count),
                sample_count=int(count),
            ))

    # -- posterior and information solves (item 3) ----------------------------------------------------------------
    def _dual_solve(self, models: DualModels, right: Any, duals: Any, residual: Any, read: _Read, column_models: np.ndarray, bound: Any) -> tuple[Any, Any, _Read]:
        """S z = right to ``bound`` per column from (duals, their exact residual): CG without its certifying read, then
        one fused read that gives the exact residual and X~'z (every column kept)."""
        array_module = self.array_module
        if not bool(array_module.any(array_module.linalg.norm(residual, axis=0) > bound)):
            return duals, residual, read
        correction = certified_block_cg(
            self.source, models, residual, array_module.zeros_like(residual), array_module.asarray(column_models), bound, self.count,
            deflation=Deflation({}, {}, {}, self._resolved), label="posterior", confirm=False,
        )
        duals = duals + correction.solution
        image, _prior, kept, _columns = self._read(models, values=duals, column_models=column_models, keep=np.arange(int(duals.shape[1])), label="posterior-fused")
        return duals, right - self._operator_image(models, duals, image, column_models), _Read(duals, image, kept)

    def _new_state(self, right: Any, model: int) -> PosteriorState:
        values, bulk_values, _resolved, column_models, image = self._bulk_image(right, model)
        rhs = -image
        zero = self.array_module.zeros_like(rhs)
        return PosteriorState(values, bulk_values, _host(column_models), rhs, zero, rhs.copy(), _Read(zero, zero, self.array_module.zeros((self.source.variant_count, int(rhs.shape[1])))), self.operator)

    def posterior_solve_state(self, right: Any, model: int, error_bound: Any, state: PosteriorState | None = None) -> tuple[Any, PosteriorState]:
        """``DualGaussian.posterior_solve`` with the fused end, continued from ``state`` (same right-hand side, same
        sites) when given; the solution and the state to continue from."""
        array_module = self.array_module
        solver_state = self._state
        models = solver_state["models"]
        bulk_variances = solver_state["bulk_variances"][:, model]
        if state is None or state.operator != self.operator:
            state = self._new_state(right, model)
        resolved = array_module.asarray(self._resolved[model])
        columns = int(state.values.shape[1])
        target = array_module.broadcast_to(array_module.asarray(error_bound, dtype=array_module.float64), (columns,)).copy()
        bound = target.copy()
        block = solver_state["blocks"].get(model)
        design_products = self._products.get(model)
        duals, residual, read = state.duals, state.residual, state.read
        while True:
            duals, residual, read = self._dual_solve(models, state.right_hand_side, duals, residual, read, state.column_models, bound)
            if block is None:
                certificate, resolved_values = array_module.linalg.norm(residual, axis=0), None
            else:
                resolved_values, _mean_duals, certificate = split_columns(array_module, block, state.values[resolved], duals, residual)
            open_mask = _host(certificate > target)
            if not open_mask.any():
                break
            finite = array_module.isfinite(certificate)
            fallback = target / array_module.maximum(array_module.linalg.norm(state.right_hand_side, axis=0), np.finfo(np.float64).tiny)
            shortfall = array_module.where(finite, target / array_module.where(finite, certificate, 1.0), fallback)
            open_columns = array_module.asarray(np.flatnonzero(open_mask))
            bound[open_columns] *= shortfall[open_columns]
            if block is not None:
                tightening = float(array_module.min(shortfall[open_columns]))
                resolved_columns = array_module.full(int(block.design.shape[1]), model)
                resolved_bound = tightening * array_module.linalg.norm(block.residual, axis=0)
                refined = certified_block_cg(self.source, models, block.design, block.duals, resolved_columns, resolved_bound, self.count, deflation=Deflation({}, {}, {}, self._resolved), label="posterior-resolved")
                block = resolved_block(array_module, block.design, block.precision, refined.solution, refined.residual)
                solver_state["blocks"][model] = block
                # The kept X~'Z_L belongs to the old Z_L: the refined one's products come from one read.
                design_products = None
        products = read.products
        if block is not None:
            if design_products is None:
                left = models.sample_to_design(block.duals, array_module.full(int(block.duals.shape[1]), model))
                design_products = array_module.empty((self.source.variant_count, int(block.duals.shape[1])))
                for start, stop, tile in self.source.blocks():
                    design_products[start:stop] = tile.rmatmat(left)
                self.count.note(int(block.duals.shape[1]), 0.0, "posterior-resolved-products")
                self._products[model] = design_products
            products = products - design_products @ resolved_values
        solution = bulk_variances[:, None] * (state.bulk_values + products)
        if resolved_values is not None:
            solution[resolved] = resolved_values
        state.duals, state.residual, state.read = duals, residual, read
        return solution, state

    def posterior_solve(self, right: Any, model: int, error_bound: Any) -> Any:
        return self.posterior_solve_state(right, model, error_bound)[0]

    def information_solve(self, probes: Any, model: int, residual_tolerance: Any) -> tuple[Any, Any, Any]:
        """``DualGaussian.information_solve`` with the fused end: the certifying read also gives Xt'w."""
        array_module = self.array_module
        solver_state = self._state
        models = solver_state["models"]
        state = self._new_state(probes, model)
        image = -state.right_hand_side
        image_norms = array_module.linalg.norm(image, axis=0)
        bound = array_module.broadcast_to(array_module.asarray(residual_tolerance, dtype=array_module.float64), (int(image.shape[1]),)) * image_norms
        duals, residual, read = self._dual_solve(models, image, state.duals, image.copy(), state.read, state.column_models, bound)
        block = solver_state["blocks"].get(model)
        coupling = array_module.zeros((0, int(image.shape[1])))
        back_products = read.products
        if block is not None:
            coupling = block.duals.T @ image
            weights = _cholesky_solve(array_module, block.factor, coupling - state.values[array_module.asarray(self._resolved[model])])
            back_products = back_products - self._products[model] @ weights
        return back_products, coupling, array_module.linalg.norm(residual, axis=0) / array_module.maximum(image_norms, np.finfo(np.float64).tiny)


def _posterior(
    gaussian: EconomicalDualGaussian, model: int, grams: BlockGrams, variances: F64Array, ensure: Callable[[], None]
) -> GaussianPosterior:
    """``full_data_fit._posterior`` whose relative_solve continues each tightening from the last (item 3)."""
    solve = gaussian.bulk_solves[model]

    def relative_solve(right: F64Array, relative_tolerance: float) -> F64Array:
        values = np.asarray(right, dtype=np.float64)
        solution = np.zeros_like(values)
        live = np.flatnonzero(np.any(values != 0.0, axis=0))
        bound = relative_tolerance * np.sqrt(np.square(values[:, live]).T @ variances)
        ensure()
        state: PosteriorState | None = None
        while live.size:
            solved, state = gaussian.posterior_solve_state(values[:, live], model, bound, state)
            solved = np.asarray(_host(solved), dtype=np.float64)
            lower, _upper = _norm_bounds(np.sum(values[:, live] * solved, axis=0), bound)
            done = bound <= relative_tolerance * lower
            solution[:, live[done]] = solved[:, done]
            bound = np.where(lower > 0.0, relative_tolerance * lower, 0.5 * bound)[~done]
            keep = np.flatnonzero(~done)
            state = PosteriorState(
                state.values[:, keep], state.bulk_values[:, keep], state.column_models[keep], state.right_hand_side[:, keep], state.duals[:, keep],
                state.residual[:, keep], _Read(state.read.duals[:, keep], state.read.image[:, keep], state.read.products[:, keep]), state.operator,
            )
            live = live[~done]
        return solution

    return GaussianPosterior(
        solve=relative_solve, variance_jvp=lambda weights: variance_jvp(solve, grams, weights).values, local_response=local_response(solve, grams),
    )


class EconomicalFixedPoints(_FullDataFixedPoints):
    """``full_data_fit._FullDataFixedPoints`` with items 3-5; its refresh and every check it certifies are the parent's."""

    gaussian: EconomicalDualGaussian

    def _iterate(self, site_precision: F64Array, site_shift: F64Array, with_probes: bool = True, error_bound: F64Array | None = None) -> None:
        target = np.sqrt(self.effective / self.draw_count)
        bound = target if error_bound is None else np.maximum(error_bound, target)
        reads = self.gaussian.count.passes
        certificate = self.gaussian.iterate(
            site_precision=site_precision, site_shift=site_shift, noise_variance=self.noise, error_bound=bound,
            probe_residual_ratio=self.probe_ratio, with_probes=with_probes,
        )
        self.mean_error = np.asarray(_host(certificate.error_bound), dtype=np.float64)
        # The reads of this solve that its Krylov iterations do not account for (item 5's fixed reads).
        self.fixed_reads = float(self.gaussian.count.passes - reads - int(certificate.iterations))
        self.passes += 1
        self.version += 1

    def _snapshot(self) -> dict:
        snapshot = super()._snapshot()
        snapshot["solver"] = self.gaussian.save()
        return snapshot

    def _restore(self, snapshot: dict) -> None:
        """The oracle and the dual solver back at a snapshot's sites, from the solver state kept with it (item 4)."""
        self.site_precision, self.site_shift = snapshot["site_precision"].copy(), snapshot["site_shift"].copy()
        self.noise, self.effective = snapshot["noise"].copy(), snapshot["effective"].copy()
        self.probe_ratio = snapshot["probe_ratio"]
        self.gaussian.load(snapshot["solver"])
        self.version += 1
        snapshot["version"] = self.version

    def _refresh(self, hyperparameters: Sequence[MixtureHyperparameters]):
        variances, grams = super()._refresh(hyperparameters)
        self._last_variances, self._last_grams = variances, grams
        return variances, grams

    def _move_bounds(self, model: int, right: F64Array, threshold: float) -> float:
        """The parent's decision of ||Sigma right||_A^2 against ``threshold``, each halving continued (item 3)."""
        bound = np.array([0.5 * np.sqrt(threshold)])
        state: PosteriorState | None = None
        while True:
            solved, state = self.gaussian.posterior_solve_state(right[:, None], model, bound, state)
            solved = np.asarray(_host(solved), dtype=np.float64)
            lower, upper = _norm_bounds(np.array([float(right @ solved[:, 0])]), bound)
            if upper[0] * upper[0] <= threshold or lower[0] * lower[0] > threshold:
                return float(upper[0] * upper[0])
            bound = 0.5 * bound

    def _solve(self, hyperparameters: Sequence[MixtureHyperparameters]) -> list[FixedPoint]:
        points = super()._solve(hyperparameters)
        # The parent's posteriors use full_data_fit._posterior; each is rebuilt with continuation and a solver snapshot.
        snapshot = self._snapshot()
        return [
            FixedPoint(
                cavity=point.cavity,
                posterior=_posterior(self.gaussian, model, self._last_grams[model], self._last_variances[:, model], lambda snapshot=snapshot: self._ensure(snapshot)),
                mean=point.mean, precision_norm=point.precision_norm, effective_effects=point.effective_effects,
            )
            for model, point in enumerate(points)
        ]

    def _frozen_passes(self, hyperparameters: Sequence[MixtureHyperparameters], frozen: F64Array, target_precision: F64Array, target_shift: F64Array) -> None:
        """The parent's mean-only EP (full_data_fit's step 1c) with item 5: no probe columns, and every pass but the
        ones that decide the exit solved to the forcing accuracy; the exit is tested only on a pass at the
        certificate's own bound."""
        gaussian = self.gaussian
        model_count = gaussian.model_count
        previous_move, damping = np.full(model_count, np.inf), 1.0
        forcing = np.zeros(model_count)
        while True:
            mean = np.asarray(_host(gaussian.mean), dtype=np.float64).copy()
            fraction = damping
            move = max(float(np.max(np.abs(target_precision - self.site_precision))), float(np.max(np.abs(target_shift - self.site_shift))))
            scale = 1.0 + max(float(np.max(np.abs(self.site_precision))), float(np.max(np.abs(self.site_shift))))
            finite = np.isfinite(previous_move)
            loose = np.where(finite & (forcing > 0.0), forcing * np.sqrt(np.where(finite, previous_move, 0.0)), 0.0)
            while True:
                if fraction * move <= _EPSILON * scale:
                    raise NoFixedPoint("no damped EP pass keeps the full-data precision positive definite")
                trial_precision = self.site_precision + fraction * (target_precision - self.site_precision)
                trial_shift = self.site_shift + fraction * (target_shift - self.site_shift)
                try:
                    self._iterate(trial_precision, trial_shift, with_probes=False, error_bound=loose)
                    break
                except np.linalg.LinAlgError:
                    fraction *= 0.5
            self.site_precision, self.site_shift = trial_precision, trial_shift
            marginal = 1.0 / (frozen + self.site_precision)
            mean_move = np.sum(np.square(_host(gaussian.mean) - mean) / marginal, axis=0) / (fraction * fraction)
            exact = bool(np.all(loose <= np.sqrt(self.effective / self.draw_count)))
            if exact and np.all(mean_move <= self.effective / self.draw_count):
                return
            ratio = mean_move / previous_move
            if float(np.max(ratio)) >= 1.0:
                damping = min(damping, 1.0 / (1.0 + np.sqrt(float(np.max(ratio)))))
            contraction = np.sqrt(np.where(np.isfinite(ratio), ratio, np.nan))
            previous_move = mean_move
            # The next pass decides the exit when the move it predicts (rho^2 times this one) is below the threshold:
            # it is then solved to the certificate's bound; otherwise to the forcing accuracy.
            predicted = np.where(np.isfinite(contraction), contraction * contraction * mean_move, np.inf)
            forcing = np.array([
                0.0 if predicted[model] <= self.effective[model] / self.draw_count or not np.isfinite(contraction[model])
                else optimal_forcing(float(contraction[model]), gaussian.measured_rate, self.fixed_reads)
                for model in range(model_count)
            ])
            new_mean = np.asarray(_host(gaussian.mean), dtype=np.float64)
            cavities = [
                Cavity(precision=frozen[:, model], shift=new_mean[:, model] / marginal[:, model] - self.site_shift[:, model]) for model in range(model_count)
            ]
            target_precision, target_shift = self._targets(hyperparameters, cavities)


def fit_full_data(
    *, gaussian: EconomicalDualGaussian, statistics: Any, prior: ScaleMixturePrior, draw_count: int, working_bytes: int, seed: int
) -> FullDataFit:
    """``full_data_fit.fit_full_data`` with ``EconomicalFixedPoints``: the same starts, certificate and answers."""
    moments = moment_starts(statistics, prior)
    if len(moments) != gaussian.model_count:
        raise ValueError("Stage 0's targets must be the models, in order")
    starts = [initial_hyperparameters(prior, moment.mean_variance) for moment in moments]
    fixed_points = EconomicalFixedPoints(gaussian, statistics, prior, draw_count, working_bytes, seed, starts, np.array([moment.noise for moment in moments]))
    try:
        fits = fit_hyperparameters(prior, starts, fixed_points, working_bytes, 0.5 / draw_count)
    except FloatingPointError as error:
        raise FloatingPointError(f"{error}; EP refusals: {fixed_points.refusals}") from error
    return FullDataFit(
        gaussian=gaussian,
        site_precision=fixed_points.site_precision,
        site_shift=fixed_points.site_shift,
        hyperparameters=tuple(fit.hyperparameters for fit in fits),
        noise_variance=fixed_points.noise,
        certificate=FitCertificate(
            remaining_gain=np.array([fit.remaining_gain for fit in fits]),
            newton_decrement=np.array([fit.newton_decrement for fit in fits]),
            smoothing_gradient=np.array([fit.step.smoothing_gradient for fit in fits]),
            stationarity_steps=tuple(fit.step.stationarity_steps for fit in fits),
            stationarity_errors=tuple(fit.step.stationarity_errors for fit in fits),
            mean_move=fixed_points.mean_move,
            draw_tolerance=fixed_points.effective / draw_count,
            noise_gain=fixed_points.noise_gain,
            mean_error=fixed_points.mean_error,
            information_bound=np.array([float(np.max(certificate.upper_bound)) for certificate in fixed_points.information]),
            information_tolerance=np.array([float(np.min(certificate.tolerance)) for certificate in fixed_points.information]),
            undecided_blocks=fixed_points.undecided_blocks,
            negative_sites=np.sum(fixed_points.site_precision < 0.0, axis=0).astype(np.int64),
            effective_effects=fixed_points.effective,
            outer_iterations=np.array([fit.iterations for fit in fits], dtype=np.int64),
            halvings=np.array([fit.halvings for fit in fits], dtype=np.int64),
            prediction_move=np.array([fit.prediction_move for fit in fits]),
            prediction_tolerance=np.array([fit.prediction_tolerance for fit in fits]),
            unresolved=np.array([fit.unresolved for fit in fits], dtype=np.int64),
            refusals=tuple(fixed_points.refusals),
            outer_history=tuple(fit.history for fit in fits),
            refreshes=fixed_points.refreshes,
            passes=fixed_points.passes,
        ),
    )
