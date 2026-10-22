"""A fitted model as the start of another fit: the empirical Bayes warm start across runs, code versions and arms.

A whole-genome arm spends hours in empirical Bayes (EB: 1-5 h on the All of Us arms [real]), and a code fix, a new
round, a deleted job or a new run identity started every one of them from the prior's moment start again: no EB state
outlived its run (22 arm fits were deleted 1.6 h in, and earlier rounds that reached EB were each restarted by code
fixes [real]). A saved model (``artifact.save_model``) already holds everything a start needs: each member's store row
and standardization scale (``fast_scoring.ScoringModel``), the noise variance, the hyperparameters, and, in the
mean-field route's draw law (``draw_laws.ProductMixtureDraws``), the lattice they are written on, every class's log
density there, every member's log prior scale and, per component of the mode mixture, every member's natural
parameters, from which the main fixed point's means are exact (``main_means``). So every model any run has saved, of
any code version, is a seed (``Seed``) for a later fit of the same trait: the next round's refit of the same arm, the
other arm of the same trait (with_sv's members are snv_indel's plus the structural records, on the same store rows),
or the same arm after a change of its filters.

``warm_start`` maps a seed onto a new fit's prior and members, by store row (the seed and the fit must index one
store's sites, ``artifact.sites_digest``), as the continuation ``full_data_fit.fit_full_data`` already takes from a fit
on another lattice (``starts``, ``start_noise``, ``start_mean``):

- the members' means, the main fixed point's (the state the seed's outer loop ended at, not the mode mixture's average
  the scorer reads, which is no fixed point of a single q: started from that average, a refit of the seed's own
  problem climbed 0.63 nats on its first step on the synthetic store), each a mapped member's raw effect kept (its
  standardized mean times the new scale over the seed's: the scales are the training rows' code sd, which a changed
  cohort moves) and zero on a member the seed did not fit (a structural record seeded from snv_indel);
- the noise variance, the seed's (a binary model, whose noise is known, is never seeded);
- the hyperparameters, the seed's own where they are this prior's (the same classes, by their store variant class, and
  the same schema, verified by the class densities and every shared member's log prior scale they give, which must be
  the seed's to rounding; carried to this lattice by ``transfer_hyperparameters`` where the seed's lattice is another),
  else the density alone: the seed's class density with the most shared members, continued onto this lattice as the
  pooled density, every class deviation, annotation effect and level zero at its lambda = infinity edge, from which the
  EB releases each block only where its evidence rises, as from the nested start (``embed_hyperparameters``).

What a seed changes, and what it cannot. The fit from a seed is the same EB outer loop on the same prior and data,
with the same acceptance criteria (the remaining gain within 1 / (2K) nats, the prediction move within p_eff / K) and
a certificate computed at the state it returns: no seed value is ever reported as a fit. A seed only moves where the
loop starts, and skips the nested search from the moment start. At a seed that is this problem's own fixed point the
loop's first check finds it stationary, and its mode mixture solves the saved components again, its settled search
standing where its main fixed point comes back (``full_data_fit._mode_mixture``: no ridge start or order from zero;
on the synthetic store of ``tests/test_warm_seeds.py`` the refit's passes over the data fell from 1086 to 78, 72 of
them the solves of the saved mixture's 18 other components [sim]). From any other seed it climbs to a fixed point
like any start; the EB objective has local maxima, so it can be another one than the cold start reaches, within the
criteria, which is why the note says where each model started (``note``, among the certificate's refusals, where
``model_acceptance`` reads a note as information). For the same reason where a fit starts is in its keys: the fit
step's holds the configured patterns (``workspace_pipeline.WorkspaceConfig.warm_start_models``), and each seed carries
its saved model's digest (``digest``, its model.json's, which holds its arrays' checksum) for the engine's checkpoints,
so a fit's checkpoints are never resumed from another start.

What a seed cannot recover: Stage 0, the summary band and the resident codes are the run's own (a seed holds no LD),
and a binary model's sites, an EP fit's or a model with no member are no seed (``seeds_from_model``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.artifact import _METADATA, FittedModel, load_model, named_digest
from sv_pgs.config import TraitType
from sv_pgs.draw_laws import ProductMixtureDraws, tile_rows
from sv_pgs.progress import log
from sv_pgs.scale_mixture_ep import (
    MixtureHyperparameters,
    ScaleMixturePrior,
    _components,
    class_log_density,
    log_scale,
    scale_mixture_prior,
    transfer_hyperparameters,
)

_ROUNDING = float(np.sqrt(np.finfo(np.float64).eps))
"""The agreement of two evaluations of one prior's densities and scales from the same coefficients, written and read
back exactly (float64 NPZ): only their own rounding, which sqrt(eps) relative bounds with room."""


@dataclass(frozen=True)
class Seed:
    """One saved model's state, as ``warm_start`` reads it: ``source`` (where it was read, for the note), the model's
    ``model_name``; per member (sorted ``store_rows`` of the store whose sites are ``sites_digest``): the training rows'
    code sd ``signed_scales``, the main fixed point's posterior mean ``means`` of the standardized column (``main_means``), the prior ``class_index`` and
    log prior scale ``log_scale`` (log u_j); per class the lattice's ``log_density`` on ``log_variance_grid``; and the
    model's ``hyperparameters`` and ``noise``. ``problem_digest`` and ``offset_digest`` say which fitting problem and
    which records' reliabilities it was fitted on (``artifact.Provenance``); ``digest`` is the SHA-256 of its saved
    model.json (``read_seeds``), empty for a model read from memory."""

    source: str
    model_name: str
    store_rows: I64Array
    signed_scales: F64Array
    means: F64Array
    class_index: I64Array
    log_scale: F64Array
    log_density: F64Array
    log_variance_grid: F64Array
    hyperparameters: MixtureHyperparameters
    noise: float
    sites_digest: str
    problem_digest: str
    offset_digest: str
    # The saved mode mixture's components' means (components x members, the main fixed point's first), their noise
    # variances (each fixed point's own, the first ``noise``: ``component_noises``) and whether its search settled with
    # nothing ended at its budget (``full_data_fit.MixtureStart``); none and False where unknown.
    components: F64Array | None = None
    component_noises: F64Array | None = None
    settled: bool = False
    digest: str = ""

    def __post_init__(self) -> None:
        rows = np.asarray(self.store_rows)
        members = rows.shape[0]
        if rows.ndim != 1 or rows.dtype != np.int64 or members == 0 or rows[0] < 0 or np.any(np.diff(rows) <= 0):
            raise ValueError("a seed's store rows must be sorted, distinct and non-negative.")
        for name in ("signed_scales", "means", "log_scale"):
            values = np.asarray(getattr(self, name))
            if values.shape != (members,) or values.dtype != np.float64 or not np.all(np.isfinite(values)):
                raise ValueError(f"a seed's {name} must be finite float64, one per member.")
        if np.any(np.asarray(self.signed_scales) <= 0.0):
            raise ValueError("a seed's scales must be positive.")
        density = np.asarray(self.log_density)
        grid = np.asarray(self.log_variance_grid)
        classes = np.asarray(self.class_index)
        if grid.ndim != 1 or grid.shape[0] < 2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
            raise ValueError("a seed's lattice must be increasing finite nodes.")
        if density.ndim != 2 or density.shape[1] != grid.shape[0] or np.any(np.isnan(density)):
            raise ValueError("a seed's log densities must be classes x nodes.")
        if classes.shape != (members,) or classes.dtype != np.int64 or np.any(classes < 0) or np.any(classes >= density.shape[0]):
            raise ValueError("a seed's class index must name one of its classes per member.")
        coefficients = np.asarray(self.hyperparameters.coefficients)
        if coefficients.ndim != 1 or not np.all(np.isfinite(coefficients)) or np.any(np.isnan(self.hyperparameters.log_smoothing)):
            raise ValueError("a seed's hyperparameters must be finite coefficients and non-NaN log penalty weights.")
        if not np.isfinite(self.noise) or self.noise <= 0.0:
            raise ValueError("a seed's noise variance must be positive.")
        if self.components is not None:
            components = np.asarray(self.components)
            if components.ndim != 2 or components.shape[0] == 0 or components.shape[1] != members or components.dtype != np.float64 or not np.all(np.isfinite(components)):
                raise ValueError("a seed's mixture components must be finite float64, components x members.")
            noises = np.asarray(self.component_noises if self.component_noises is not None else np.zeros(0))
            if noises.shape != components.shape[:1] or not np.all(np.isfinite(noises)) or not np.all(noises > 0.0):
                raise ValueError("a seed's mixture components each need a positive noise variance.")


def content_digest(seed: Seed) -> str:
    """SHA-256 of everything a fit takes from ``seed`` (``warm_start``, whose note names its source and model): its
    members, scales, means, classes, log scales, lattice and densities, hyperparameters, noise, mixture components and
    their noises, whether its search settled, and the store sites, problem and offsets it was fitted on. A fit's
    checkpoints name a seed read from memory by it (``fit_model.fit_identity``): such a seed has no saved model's
    ``digest``, and its source alone named two different seeds alike, so one's completed fit was returned for the other."""
    parts: dict[str, object] = {
        "source": seed.source, "model_name": seed.model_name, "store_rows": seed.store_rows, "signed_scales": seed.signed_scales,
        "means": seed.means, "class_index": seed.class_index, "log_scale": seed.log_scale, "log_density": seed.log_density,
        "log_variance_grid": seed.log_variance_grid, "coefficients": seed.hyperparameters.coefficients,
        "log_smoothing": seed.hyperparameters.log_smoothing, "noise": np.array([seed.noise]), "settled": str(seed.settled),
        "sites_digest": seed.sites_digest, "problem_digest": seed.problem_digest, "offset_digest": seed.offset_digest,
    }
    if seed.components is not None:
        parts["components"] = seed.components
        parts["component_noises"] = np.asarray(seed.component_noises, dtype=np.float64)
    return named_digest(parts)


def for_problem(seed: Seed, problem: str, offset: str, sites: str) -> Seed:
    """``seed`` as a start of the fit of ``problem`` and ``offset`` (``artifact.problem_digest``, ``offset_digest``) on
    a store whose sites are ``sites``: its mode search's ``settled`` stands only where all three are the seed's own. The
    search settled on the modes of its own posterior; another cohort, phenotype, split, covariates or records'
    reliabilities move that landscape, where the seed's components can each still be a fixed point while a mode its
    search never met is the higher one, so the search runs again (the components still start the mixture:
    ``full_data_fit.MixtureStart``)."""
    own = seed.problem_digest == problem and seed.offset_digest == offset and seed.sites_digest == sites
    return seed if own or not seed.settled else replace(seed, settled=False)


def main_means(law: ProductMixtureDraws, working_bytes: int, component: int = 0) -> F64Array:
    """Each member's posterior mean in the law's first component, the main fixed point (``full_data_fit._mode_mixture``
    puts it first): over the lattice nodes, the node responsibilities' average of h c_k, the mean of the N(h c_k, c_k)
    the law draws from at node k (``draw_laws.ProductMixtureDraws.tile``), at the component's natural parameters
    (h, omega). It is the main fixed point's q mean exactly, which the mixture's mean (the scorer's coefficients)
    averages with the other modes. A tile of rows at a time within ``working_bytes``. Any other ``component``'s means
    likewise, the other modes' fixed points (the mode mixture's start in a fit from this seed)."""
    count = law.shape[0]
    means = np.empty(count)
    step = tile_rows(law, working_bytes)
    grid = np.asarray(law.log_variance_grid, dtype=np.float64)
    classes = np.asarray(law.class_index)
    for start in range(0, count, step):
        rows = np.arange(start, min(start + step, count))
        for position in np.unique(classes[rows]):
            members = rows[classes[rows] == position]
            shift = np.asarray(law.shift[component], dtype=np.float64)[members]
            terms = _components(
                np.asarray(law.log_density[position], dtype=np.float64), np.asarray(law.log_scale, dtype=np.float64)[members], grid,
                np.asarray(law.omega[component], dtype=np.float64)[members], shift,
            )
            means[members] = shift * np.sum(np.asarray(terms.responsibility) * np.asarray(terms.conditional_variance), axis=1)
    return means


def component_noise(law: ProductMixtureDraws, component: int, noise: float) -> float:
    """The noise variance of the law's ``component``, from the first's (``noise``): each component's omega is its
    members' squared projected columns over its own noise (``full_data_fit._mode_mixture``), so omega_0 / omega_c is
    sigma_c^2 / sigma_0^2 at every member; the median over the members where both are positive, the first's own where
    none is."""
    first, other = np.asarray(law.omega[0], dtype=np.float64), np.asarray(law.omega[component], dtype=np.float64)
    usable = (first > 0.0) & (other > 0.0)
    return float(noise) * float(np.median(first[usable] / other[usable])) if usable.any() else float(noise)


def seeds_from_model(model: FittedModel, source: str, working_bytes: int, digest: str = "") -> dict[str, Seed]:
    """Each model of ``model`` that can seed a fit, by its name: a quantitative model fitted by the mean-field route
    (its draw law the product mixture's, which holds its lattice, densities, log scales and its main fixed point's
    natural parameters) with at least one member. A binary model (its start is its Polya-Gamma sites, which no artifact
    holds), an EP or tied fit (a dense law) and the null genetic model (no member) are none."""
    seeds: dict[str, Seed] = {}
    # Each model's count of components (the law pads a group's shorter mixtures with its first at weight zero), whether
    # nothing of its fit ended at a work budget, its mode search included (``FitCertificate.budget_unresolved``), and
    # whether its search settled under the whole-budget rule (``FitCertificate.mixture_settled``; a model saved before
    # that rule has no such term, and its search, stopped at the first order that moved nothing, is not kept).
    counts = np.asarray(model.certificate.get("mixture_components", np.zeros(len(model.model_names))), dtype=np.int64)
    budgeted = np.asarray(model.certificate.get("budget_unresolved", np.ones(len(model.model_names))), dtype=np.int64)
    whole = np.asarray(model.certificate.get("mixture_settled", np.zeros(len(model.model_names))), dtype=bool)
    for index, (name, scoring, hyperparameters, noise) in enumerate(zip(model.model_names, model.scoring, model.hyperparameters, np.asarray(model.noise_variance))):
        law = scoring.posterior_draws
        if scoring.trait_type != TraitType.QUANTITATIVE or getattr(law, "kind", None) != "product_mixture" or scoring.store_rows.shape[0] == 0:
            continue
        count = int(counts[index])
        components = np.stack([main_means(law, working_bytes, component) for component in range(count)]) if 0 < count <= len(law.shift) else None
        seeds[name] = Seed(
            source=source, model_name=name, store_rows=np.asarray(scoring.store_rows, dtype=np.int64),
            signed_scales=np.asarray(scoring.signed_scales, dtype=np.float64), means=main_means(law, working_bytes) if components is None else components[0].copy(),
            class_index=np.asarray(law.class_index, dtype=np.int64), log_scale=np.asarray(law.log_scale, dtype=np.float64),
            log_density=np.asarray(law.log_density, dtype=np.float64), log_variance_grid=np.asarray(law.log_variance_grid, dtype=np.float64),
            hyperparameters=hyperparameters, noise=float(noise), sites_digest=model.provenance.sites_digest,
            problem_digest=model.provenance.problem_digest, offset_digest=model.provenance.offset_digest,
            components=components, settled=components is not None and int(budgeted[index]) == 0 and bool(whole[index]), digest=digest,
            component_noises=None if components is None else np.array([component_noise(law, component, float(noise)) for component in range(count)]),
        )
    return seeds


def read_seeds(directory: str | Path, working_bytes: int) -> dict[str, Seed]:
    """The seeds of the saved model at ``directory`` (``seeds_from_model``, within ``working_bytes``), or none, with
    the reason logged, where it is not a whole saved model (``artifact.load_model`` checks its arrays' checksum)."""
    try:
        digest = hashlib.sha256((Path(directory) / _METADATA).read_bytes()).hexdigest()
        return seeds_from_model(load_model(directory), str(directory), working_bytes, digest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        log(f"warm seeds: {directory} is no seed ({error})")
        return {}


@dataclass(frozen=True)
class WarmStart:
    """A model's start from a seed (``warm_start``): ``fit_full_data``'s continuation (``hyperparameters`` on the fit's
    prior, the ``noise`` variance and the members' ``mean``) and the ``note`` its certificate carries."""

    hyperparameters: MixtureHyperparameters
    noise: float
    mean: F64Array
    note: str
    # The seed's mixture components mapped as ``mean`` (components x this fit's members), their noise variances and
    # whether its search settled (``full_data_fit.MixtureStart``); None where the seed holds none.
    components: F64Array | None = None
    component_noises: F64Array | None = None
    settled: bool = False


def _class_labels(rows: I64Array, class_index: I64Array, class_count: int, variant_class: np.ndarray) -> tuple[int, ...]:
    """Each class's store variant class (the ``dosage_store.VARIANT_CLASSES`` index its members carry), in class order."""
    return tuple(int(variant_class[rows[np.flatnonzero(class_index == position)[0]]]) for position in range(class_count))


def seed_lattice(seeds: list[Seed | None], nodes: F64Array) -> F64Array:
    """The lattice a group whose every model has a seed fits on: the seeds' one lattice where they share it and it
    covers ``nodes`` (the derived start lattice) at no wider a spacing, so the quadrature the seeds' fits refined or
    extended to is not made again; else ``nodes``. A lattice the start's covers and refines is checked at the fitted
    state like any other (``scale_mixture_ep.lattice_check``)."""
    start = np.asarray(nodes, dtype=np.float64)
    if not seeds or any(seed is None for seed in seeds):
        return start
    grid = seeds[0].log_variance_grid
    if any(not np.array_equal(seed.log_variance_grid, grid) for seed in seeds):
        return start
    spacing, start_spacing = float(grid[1] - grid[0]), float(start[1] - start[0])
    covered = float(grid[0]) <= float(start[0]) + _ROUNDING * start_spacing and float(grid[-1]) >= float(start[-1]) - _ROUNDING * start_spacing
    if not covered or spacing > start_spacing * (1.0 + _ROUNDING):
        return start
    return np.asarray(grid, dtype=np.float64).copy()


def _schema_prior(prior: ScaleMixturePrior, grid: F64Array) -> ScaleMixturePrior | None:
    """``prior``'s schema on ``grid`` (``relattice``'s construction), or None where ``grid`` is no lattice of it."""
    if np.array_equal(grid, prior.log_variance_grid):
        return prior
    try:
        return scale_mixture_prior(
            class_index=prior.class_index, log_variance_offset=prior.log_variance_offset,
            annotation_design=prior.scale_design[:, : prior.scale_size - prior.level_size], annotation_groups=prior.annotation_groups,
            offset_groups=prior.offset_groups, nodes=grid, floor=prior.kernel_floor, top=prior.kernel_top,
        )
    except ValueError:
        return None


def _agree(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.allclose(first, second, rtol=_ROUNDING, atol=_ROUNDING))


def _own_prior(
    seed: Seed, prior: ScaleMixturePrior, labels: tuple[int, ...], seed_labels: tuple[int, ...], at: I64Array, seed_at: I64Array,
) -> tuple[MixtureHyperparameters, str] | None:
    """The seed's hyperparameters on ``prior`` where they are this prior's (the module docstring), else None: the same
    classes in the same order, coefficients of this schema, and on the seed's lattice the same class densities and, at
    every shared member, the same log prior scale as the seed's own (so the annotation effects and the offsets mean what
    they meant to the seed); carried to ``prior``'s lattice where it is another."""
    hyperparameters = seed.hyperparameters
    if labels != seed_labels or at.shape[0] == 0 or hyperparameters.log_smoothing.shape != (len(prior.smoothing_blocks),):
        return None
    on_grid = _schema_prior(prior, np.asarray(seed.log_variance_grid, dtype=np.float64))
    if on_grid is None or hyperparameters.coefficients.shape != (on_grid.coefficient_size,):
        return None
    if not _agree(class_log_density(on_grid, hyperparameters.coefficients), seed.log_density):
        return None
    if not _agree(log_scale(on_grid, hyperparameters.coefficients)[at], seed.log_scale[seed_at]):
        return None
    if on_grid is prior:
        return hyperparameters, "its own hyperparameters"
    return transfer_hyperparameters(on_grid, prior, hyperparameters), f"its own hyperparameters carried from its {seed.log_variance_grid.shape[0]}-node lattice"


def _single_class(nodes: F64Array, floor: float, top: float) -> ScaleMixturePrior:
    """A one-member, one-class prior on ``nodes``: the pooled lattice function's coordinates, which ``transfer_hyperparameters``
    continues from one lattice to another."""
    return scale_mixture_prior(
        class_index=np.zeros(1, dtype=np.int64), log_variance_offset=np.zeros(1), annotation_design=np.zeros((1, 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )


def _density_start(seed: Seed, prior: ScaleMixturePrior, seed_at: I64Array) -> tuple[MixtureHyperparameters, str]:
    """The seed's density alone (the module docstring): its class with the most shared members (its largest class
    where none is shared) as the pooled density, continued onto ``prior``'s lattice with the seed's pooled roughness
    weight (``scale_mixture_prior``'s first block), every other coefficient zero and every other weight at its
    lambda = infinity edge, a start the EB moves from as from the nested start."""
    counts = np.bincount(seed.class_index[seed_at] if seed_at.shape[0] else seed.class_index, minlength=seed.log_density.shape[0])
    chosen = int(np.argmax(counts))
    grid = np.asarray(seed.log_variance_grid, dtype=np.float64)
    source = _single_class(grid, prior.kernel_floor, prior.kernel_top)
    target = _single_class(np.asarray(prior.log_variance_grid, dtype=np.float64), prior.kernel_floor, prior.kernel_top)
    density = np.asarray(seed.log_density[chosen], dtype=np.float64)
    # The pooled coordinates of the centred log density in the orthonormal sum-to-zero basis (``initial_hyperparameters``).
    pooled = source.coefficient_map[: source.grid_size, : source.pooled_size].T @ (density - density.mean())
    weight = np.asarray(seed.hyperparameters.log_smoothing[:1], dtype=np.float64)
    moved = transfer_hyperparameters(source, target, MixtureHyperparameters(coefficients=pooled, log_smoothing=weight))
    coefficients = np.zeros(prior.coefficient_size)
    coefficients[: prior.pooled_size] = moved.coefficients[: target.pooled_size]
    log_smoothing = np.full(len(prior.smoothing_blocks), np.inf)
    log_smoothing[0] = weight[0]
    return MixtureHyperparameters(coefficients=coefficients, log_smoothing=log_smoothing), f"the density of its class {chosen} alone"


def warm_start(seed: Seed, prior: ScaleMixturePrior, member_rows: I64Array, scales: F64Array, variant_class: np.ndarray) -> WarmStart | None:
    """``seed`` as the start of a fit of ``prior`` over the members ``member_rows`` (Stage 0's active rows, sorted store
    rows of the store whose ``variant_class`` is given) with Stage 0's ``scales`` (the module docstring), or None, with
    the reason logged, where the seed names rows outside this store or gives a start that is not finite."""
    rows = np.asarray(member_rows, dtype=np.int64)
    if int(seed.store_rows[-1]) >= int(np.asarray(variant_class).shape[0]):
        log(f"warm seeds: {seed.source} ({seed.model_name}) names store rows this store does not have; not used")
        return None
    _shared, at, seed_at = np.intersect1d(rows, seed.store_rows, assume_unique=True, return_indices=True)
    mean = np.zeros(rows.shape[0])
    ratio = np.asarray(scales, dtype=np.float64)[at] / seed.signed_scales[seed_at]
    mean[at] = seed.means[seed_at] * ratio
    components = None
    if seed.components is not None:
        components = np.zeros((seed.components.shape[0], rows.shape[0]))
        components[:, at] = seed.components[:, seed_at] * ratio[None, :]
    labels = _class_labels(rows, prior.class_index, prior.class_count, variant_class)
    seed_labels = _class_labels(seed.store_rows, seed.class_index, seed.log_density.shape[0], variant_class)
    found = _own_prior(seed, prior, labels, seed_labels, at, seed_at)
    hyperparameters, how = found if found is not None else _density_start(seed, prior, seed_at)
    if not np.all(np.isfinite(hyperparameters.coefficients)) or not np.all(np.isfinite(mean)) or (components is not None and not np.all(np.isfinite(components))):
        log(f"warm seeds: {seed.source} ({seed.model_name}) gives no finite start on this prior; not used")
        return None
    note = (
        f"warm start: {seed.model_name} of {seed.source} ({at.shape[0]:,} of this fit's {rows.shape[0]:,} members mapped; "
        f"{how}; its noise variance {seed.noise:.6g})"
    )
    log(f"warm seeds: {note}")
    # The seed's settled mode search stands for this fit only where the fit is the seed's own problem as far as the start
    # can tell (every member mapped both ways, on its own prior; ``for_problem`` has cleared it for another problem):
    # another arm's adds or drops columns, and the near duplicates among them are what the mixture's modes are, so its
    # search runs again (``full_data_fit._mode_mixture``).
    own = found is not None and at.shape[0] == rows.shape[0] == seed.store_rows.shape[0]
    return WarmStart(
        hyperparameters=hyperparameters, noise=float(seed.noise), mean=mean, note=note, components=components,
        component_noises=None if components is None else np.asarray(seed.component_noises, dtype=np.float64).copy(), settled=seed.settled and own,
    )
