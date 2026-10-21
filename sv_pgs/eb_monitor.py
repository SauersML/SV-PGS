"""The empirical-Bayes trajectory monitor: at each call of a mean-field fit's outer loop, the posterior mean it solved
scored on each model's evaluation rows (its held-out fold), logged beside the call's ELBO and how far its
hyperparameters moved since the call before. A fit's held-out signal then shows hours before the fit ends, and whether
it still rises once the ELBO and the hyperparameters have nearly settled says how many outer iterations are worth
their cost.

Monitoring only. Nothing in the fit reads what it computes: not the outer loop's steps, their acceptance or when it
stops, not the mode mixture, the lattice check, the certificate, the saved model or its checkpoint key. The log is its
only output, and a fit with the monitor is the fit without it, bit for bit (tests/test_eb_monitor.py), wherever the
host ledger holds the fit's caches beside the monitor's codes (``EvaluationCodes``: the one way it can touch a fit is
the room they take). The evaluation rows' targets reach the fit for it alone (``fit_model.FitRequest.evaluation``),
enter no key, checkpoint or saved model, and leave it only as aggregates: the genetic score's partial r^2 against the
trait's covariates (the report's ``pgs_partial_r2``) and, for a disease, its AUC (the report's ``pgs_auc``), beside
the call's ELBO and step norm. No row, count or case count is logged.

Its genotypes cost no read of the store. Stage 0's pass reads every sample of every candidate record
(``genotype_statistics.DosageStoreTileSource``), and ``training_codes.CodeTee`` hands each tile to ``EvaluationCodes``,
which keeps its evaluation columns as the pass goes (candidates x evaluation rows, int8 on the host, charged to the
ledger; where the ledger cannot admit them the fit runs without a monitor). A fit whose Stage 0 did not read every
member's record has no codes: the monitor says so and scores nothing.

A score is one pass over those codes with the members' weights m_j / s_j (the posterior mean over each member's
training scale, Stage 0's), a Stage 0 block cap of records at a time in float32, every model of the call at once. Its
time is held to MONITOR_SHARE of the fit's time since the monitor began: a call is scored only while the scores so far
and the last one's time fit that share, so a score slower than the share of a call is taken every few calls instead of
at each. Every call is logged, scored or not.

MONITOR_SWITCH set to 0 turns it off entirely (``stage2_wiring.fit_models`` then drops the evaluation rows): no codes
are kept, nothing is charged to the ledger, and nothing is scored or logged."""
from __future__ import annotations

import contextlib
import os
import time
import weakref
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sv_pgs._typing import BoolArray, F64Array, I64Array, U8Array
from sv_pgs.binary_likelihood import auc
from sv_pgs.genotype_buffers import SIGNED_CODE_OFFSET
from sv_pgs.memory_broker import HOST, current_broker
from sv_pgs.progress import log

MONITOR_SHARE = 0.05
"""The monitor's time as a share of the fit's at most (the lead's specification of the monitor, which no result
reads): one score more is taken only while the scores so far and the last one's time fit it."""

MONITOR_SWITCH = "SVPGS_EB_MONITOR"
"""The environment variable that, set to 0, turns the monitor off entirely (``monitor_on``); unset or anything else,
it is on. No fit result depends on it, so no step's key reads it."""


def monitor_on() -> bool:
    """Whether the monitor runs (``MONITOR_SWITCH``)."""
    return os.environ.get(MONITOR_SWITCH, "").strip() != "0"


@dataclass(frozen=True)
class EvaluationRows:
    """Each model's evaluation rows (its held-out fold), for the monitor alone (the module docstring): the store columns
    [m] of the rows any model is evaluated on, distinct; their covariates [m, k] in the request's columns (without the
    intercept in ``fit_model.FitRequest``, with it first in ``stage2_wiring.fit_models``, as the training covariates);
    the targets [m, models], NaN on a row that is not that model's evaluation row; and per model whether its outcome is
    a disease's 0/1 status (its AUC is logged beside the partial r^2), whatever likelihood the fit gives it."""

    store_columns: I64Array
    covariates: F64Array
    targets: F64Array
    disease: BoolArray

    def checked(self, store_samples: int, covariate_count: int, model_count: int) -> EvaluationRows:
        """These rows checked against a request's store, covariates and models, and converted (``FitRequest``)."""
        columns = np.asarray(self.store_columns)
        if columns.ndim != 1 or columns.dtype.kind not in "iu" or not columns.size:
            raise ValueError("evaluation store_columns must be a nonempty 1-D integer array of distinct store samples.")
        columns = columns.astype(np.int64)
        if np.unique(columns).shape[0] != columns.shape[0] or np.any(columns < 0) or np.any(columns >= store_samples):
            raise ValueError("evaluation store_columns must name distinct store samples.")
        covariates = np.asarray(self.covariates)
        if covariates.shape != (columns.shape[0], covariate_count) or covariates.dtype.kind not in "fiu" or not np.all(np.isfinite(covariates)):
            raise ValueError("evaluation covariates must be finite [evaluation rows, named covariates].")
        targets = np.asarray(self.targets)
        if targets.shape != (columns.shape[0], model_count) or targets.dtype.kind not in "fiu":
            raise ValueError("evaluation targets must be [evaluation rows, models], NaN off a model's evaluation rows.")
        disease = np.asarray(self.disease)
        if disease.shape != (model_count,) or disease.dtype != np.bool_:
            raise ValueError("evaluation disease must be one bool per model.")
        targets = targets.astype(np.float64)
        if any(not np.all(np.isin(targets[np.isfinite(targets[:, model]), model], (0.0, 1.0))) for model in np.flatnonzero(disease)):
            raise ValueError("a disease's evaluation targets must be 0 or 1.")
        return EvaluationRows(store_columns=columns, covariates=covariates.astype(np.float64), targets=targets, disease=disease)

    def group(self, models: I64Array, adjusted: BoolArray, names: Sequence[str]) -> GroupEvaluation:
        """The evaluation of a fit group's ``models`` (``stage2_wiring.fit_models``): the rows any of them is evaluated
        on, in store column order, with the group's ``adjusted`` covariates; none where none of them holds a row out
        (a group of all-rows models: its calls are still logged, unscored)."""
        targets = self.targets[:, models]
        rows = np.flatnonzero(np.isfinite(targets).any(axis=1))
        rows = rows[np.argsort(self.store_columns[rows], kind="stable")]
        return GroupEvaluation(
            store_columns=self.store_columns[rows], covariates=self.covariates[rows][:, adjusted], targets=targets[rows],
            disease=self.disease[models], names=tuple(names),
        )


@dataclass(frozen=True)
class GroupEvaluation:
    """One fit group's evaluation rows (``EvaluationRows.group``): store columns [m] ascending, the group's covariates
    [m, k] (intercept first), targets [m, group models] (NaN off a model's rows), and per model its disease flag and
    its name for the log."""

    store_columns: I64Array
    covariates: F64Array
    targets: F64Array
    disease: BoolArray
    names: tuple[str, ...]


class EvaluationCodes:
    """The evaluation rows' signed codes of Stage 0's candidate records [candidates, evaluation rows], int8 on the host,
    kept from Stage 0's own tiles (``training_codes.CodeTee``) with which candidates a tile has filled, and a float32 buffer of a
    Stage 0 block cap of them, the scores' working rows. Both are a cache of the ledger (``memory_broker``): admitted
    only from what the host has left, and evicted (the monitor then off) whenever a lease of the fit needs the room, so
    no allocation of the fit fails for them. They do take room from the fit's own host caches: one admitted later finds
    that much less, and a lease that evicts takes idle caches newest first, the fit's before these. On a host ledger
    near its capacity the fit may then recompute a Gram block or window terms it would have kept (the same numbers,
    later), or, on the host route only, solve a curvature system without its warm start (``full_data_fit._hold_start``:
    to the same tolerance from another start); on the device route that start is the device's. Where the ledger cannot
    admit them the constructor raises ``MemoryError``. A tile the codes cannot take (or any failure of theirs) turns
    them off; it never reaches Stage 0. The lease holds the codes only weakly and is released when they are collected,
    so a group's fit that ends early or fails leaves nothing on the ledger (an enclosing scope's ledger outlives it)."""

    def __init__(self, candidates: I64Array, columns: I64Array, chunk_rows: int) -> None:
        self.candidates = np.asarray(candidates, dtype=np.int64)
        self.columns = np.asarray(columns, dtype=np.int64)
        self.chunk_rows = max(1, min(int(chunk_rows), self.candidates.shape[0]))
        nbytes = self.candidates.shape[0] * self.columns.shape[0] + self.chunk_rows * self.columns.shape[0] * np.dtype(np.float32).itemsize
        # Why the codes are off (evicted, or a tile they could not take), or None while they are kept.
        self.off: str | None = None
        broker = current_broker()
        self._lease = None
        if broker is not None:
            held = weakref.ref(self)

            def evict() -> None:
                codes = held()
                if codes is not None:
                    codes._evicted()

            self._lease = broker.admit(HOST, nbytes, "the EB monitor's evaluation codes", evict)
            if self._lease is None:
                raise MemoryError(f"the EB monitor's evaluation codes ({nbytes / 1e9:.2f} GB) do not fit what the host ledger has left: {broker.describe()}")
            weakref.finalize(self, self._lease.release)
        self.codes = np.empty((self.candidates.shape[0], self.columns.shape[0]), dtype=np.int8)
        self.filled = np.zeros(self.candidates.shape[0], dtype=bool)
        self.working = np.empty((self.chunk_rows, self.columns.shape[0]), dtype=np.float32)

    def _evicted(self) -> None:
        """The ledger's eviction (the fit needs the room): the arrays dropped, the codes off."""
        self._drop("the ledger evicted the evaluation codes for the fit's own memory")

    def _drop(self, reason: str) -> None:
        if self.off is None:
            self.off = reason
            log(f"eb monitor: off: {reason}")
        self.codes = np.empty((0, self.columns.shape[0]), dtype=np.int8)
        self.working = np.empty((0, self.columns.shape[0]), dtype=np.float32)

    def capture(self, rows: I64Array, tile: U8Array) -> None:
        """Keep the evaluation columns of a Stage 0 tile: the codes [rows, every store sample] of the candidates
        ``rows``, consecutive among the candidates (one chromosome's tile)."""
        if self.off is not None:
            return
        try:
            codes, first = self.codes, int(np.searchsorted(self.candidates, rows[0]))
            target = slice(first, first + rows.shape[0])
            if not np.array_equal(self.candidates[target], rows):
                raise ValueError("a Stage 0 tile's rows are not consecutive candidates")
            np.subtract(tile[:, self.columns].view(np.int8), np.int8(SIGNED_CODE_OFFSET), out=codes[target])
            self.filled[target] = True
        except Exception as error:  # the monitor never fails Stage 0
            # The error's type alone: its message could carry the evaluation rows' shapes.
            self._drop(f"a Stage 0 tile could not be kept ({type(error).__name__})")

    def in_use(self) -> Any:
        """While inside, the codes are not evicted (``memory_broker.Lease.in_use``)."""
        return contextlib.nullcontext() if self._lease is None else self._lease.in_use()

    def release(self) -> None:
        """The arrays dropped and their lease released."""
        if self.off is None:
            self.off = "released"
        self._drop(self.off)
        if self._lease is not None:
            self._lease.release()


def _residualized(values: F64Array, covariates: F64Array) -> F64Array:
    """``values`` less their least-squares projection on ``covariates`` (whose span holds the intercept)."""
    return values - covariates @ np.linalg.lstsq(covariates, values, rcond=None)[0]


def partial_r2(target: F64Array, score: F64Array, covariates: F64Array) -> float:
    """The squared correlation of a score with a target after both are residualized on the covariates: the report's
    ``pgs_partial_r2`` (``workspace_pipeline._partial_r2``), repeated here so that the fit step's key does not cover the
    report (tests/test_eb_monitor.py holds the two to the same value)."""
    if not target.size or not all(np.all(np.isfinite(values)) for values in (target, score, covariates)):
        return float("nan")
    target_residual, score_residual = _residualized(target, covariates), _residualized(score, covariates)
    if np.std(target_residual) == 0.0 or np.std(score_residual) == 0.0:
        return float("nan")
    return float(np.corrcoef(target_residual, score_residual)[0, 1] ** 2)


class EbMonitor:
    """The monitor of one fit group (the module docstring): its evaluation codes at the members' records (None where
    there are none: every call is then logged without metrics), Stage 0's member means and scales, the group's
    evaluation rows, and the log's state (the calls so far, each model's last hyperparameters in the current stage, the
    scores' time). ``start`` builds it."""

    def __init__(
        self, codes: EvaluationCodes | None, positions: I64Array, means: F64Array, scales: F64Array, evaluation: GroupEvaluation,
    ) -> None:
        self._codes = codes
        self._positions = positions
        self._means = np.asarray(means, dtype=np.float64)
        self._scales = np.asarray(scales, dtype=np.float64)
        self._evaluation = evaluation
        # Which of the group's models have evaluation rows (an all-rows model holds none out: it is logged unscored).
        self._evaluated = np.isfinite(evaluation.targets).any(axis=0)
        self._calls = 0
        self._stage = ""
        self._previous: list[F64Array | None] = [None] * len(evaluation.names)
        self._began = time.perf_counter()
        self._spent = 0.0
        self._last = 0.0
        self._off = False

    @classmethod
    def start(
        cls, codes: EvaluationCodes | None, members: I64Array, means: F64Array, scales: F64Array, evaluation: GroupEvaluation,
    ) -> EbMonitor:
        """The monitor of a group over the fit's ``members`` (Stage 0's active store rows, with their training means and
        scales), scoring from the ``codes`` Stage 0's pass kept where they hold every member's record; where they do not
        (none admitted, evicted, or a Stage 0 that did not read the store) it logs why and logs every call unscored."""
        rows = np.asarray(members, dtype=np.int64)
        positions = np.zeros(0, dtype=np.int64)
        if codes is not None and codes.off is None:
            positions = np.searchsorted(codes.candidates, rows)
            if not np.all(positions < codes.candidates.shape[0]) or not np.array_equal(codes.candidates[positions], rows) or not np.all(codes.filled[positions]):
                codes.release()
                codes = None
                log("eb monitor: unscored: Stage 0's pass did not read every member's record from the store, so the evaluation rows have no codes")
        elif codes is not None:
            codes = None
        if not np.isfinite(evaluation.targets).any():
            scoring = "unscored: no model of the group holds a row out"
        elif codes is None:
            scoring = "unscored"
        else:
            scoring = f"scored from the codes Stage 0's pass kept while the scores take at most {MONITOR_SHARE:.0%} of the fit's time"
        log(f"eb monitor: on for {', '.join(evaluation.names)}: each outer-loop call logged, {scoring}")
        return cls(codes, positions, means, scales, evaluation)

    def stage(self, name: str) -> None:
        """A new stage of the fit (a new oracle: the nested fit's base or annotated prior, a continued refit): each
        model's hyperparameter move restarts there, its coefficients being another prior's."""
        self._stage = name
        self._previous = [None] * len(self._previous)

    def view(self, models: Sequence[int]) -> MonitorView:
        """The monitor as one fit of the group's ``models`` sees it (``fit_full_data``: its model k is ``models[k]``)."""
        return MonitorView(self, tuple(int(model) for model in models))

    def scores(self, means: F64Array) -> F64Array:
        """The genetic scores [evaluation rows, models] of the members' posterior means [members, models]: sum_j
        (c_ij - mu_j) / s_j m_j, Stage 0's training means and scales, the codes a block cap of candidates at a time
        (not evicted meanwhile)."""
        codes = self._codes
        assert codes is not None
        weights = np.zeros((codes.candidates.shape[0], means.shape[1]), dtype=np.float32)
        weights[self._positions] = means / self._scales[:, None]
        scores = np.zeros((codes.columns.shape[0], means.shape[1]))
        with codes.in_use():
            held, working = codes.codes, codes.working
            if held.shape[0] != codes.candidates.shape[0]:
                raise MemoryError("the evaluation codes were evicted")
            for start in range(0, held.shape[0], codes.chunk_rows):
                stop = min(start + codes.chunk_rows, held.shape[0])
                np.copyto(working[: stop - start], held[start:stop])
                scores += working[: stop - start].T @ weights[start:stop]
        return scores - self._means @ (means / self._scales[:, None])

    def observe(self, models: tuple[int, ...], oracle: Any, hyperparameters: Sequence[Any], solving: Sequence[int]) -> None:
        """One line per solved model of an oracle call (``_FullDataMeanField.__call__``): the call's ELBO, the move of
        its hyperparameters' coefficients from the model's last call in this stage (||dx||, '-' at a stage's first),
        and, where the codes are kept, the share allows and the model has evaluation rows (an all-rows model has none),
        the posterior mean's held-out metrics. A failure of the monitor turns it off and never reaches the fit."""
        if self._off:
            return
        try:
            self._observe(models, oracle, hyperparameters, solving)
        except Exception as error:  # the monitor never fails a fit: nothing reads it
            self._off = True
            # The error's type alone: its message could carry the evaluation rows' shapes.
            log(f"eb monitor: off after an error ({type(error).__name__}); the fit goes on unmonitored")

    def _observe(self, models: tuple[int, ...], oracle: Any, hyperparameters: Sequence[Any], solving: Sequence[int]) -> None:
        self._calls += 1
        now = time.perf_counter()
        codes = self._codes
        if codes is not None and codes.off is not None:
            self._codes = codes = None
        evaluated = [column for column in solving if self._evaluated[models[column]]]
        scored = codes is not None and bool(evaluated) and self._spent + self._last <= MONITOR_SHARE * (now - self._began)
        metrics = {column: "not scored (no evaluation rows)" for column in solving if column not in evaluated}
        unscored = "not scored (no evaluation codes)" if codes is None else "not scored (the monitor's share of the fit's time is spent)"
        if scored:
            started = time.perf_counter()
            scores = self.scores(np.column_stack([np.asarray(oracle.mean[:, column], dtype=np.float64) for column in evaluated]))
            for position, column in enumerate(evaluated):
                metrics[column] = self._metrics(models[column], scores[:, position])
            self._last = time.perf_counter() - started
            self._spent += self._last
        for column in solving:
            model = models[column]
            coefficients = np.asarray(hyperparameters[column].coefficients, dtype=np.float64)
            previous = self._previous[model]
            step = f"{float(np.linalg.norm(coefficients - previous)):.4g}" if previous is not None and previous.shape == coefficients.shape else "-"
            self._previous[model] = coefficients.copy()
            log(f"eb monitor: model {self._evaluation.names[model]} iter {self._calls} elbo {float(oracle.elbo[column]):.10g} "
                f"{metrics.get(column, unscored)} x_step {step} stage {self._stage}")
        if scored:
            log(f"eb monitor: scored in {self._last:.2f} s ({self._spent:.0f} s of the fit's {now - self._began:.0f} s so far)")

    def _metrics(self, model: int, score: F64Array) -> str:
        """The held-out metrics of one model's genetic score on its evaluation rows."""
        evaluation = self._evaluation
        rows = np.isfinite(evaluation.targets[:, model])
        targets, covariates = evaluation.targets[rows, model], evaluation.covariates[rows]
        text = f"pgs_partial_r2 {partial_r2(targets, score[rows], covariates):.5g}"
        if evaluation.disease[model]:
            cases = int(targets.sum())
            value = auc(targets, score[rows]) if 0 < cases < targets.shape[0] else float("nan")
            text += f" pgs_auc {value:.5g}"
        return text

    def close(self) -> None:
        """The monitor off and its codes' lease released (the group's fit has ended)."""
        self._off = True
        if self._codes is not None:
            self._codes.release()
            self._codes = None


@dataclass(frozen=True)
class MonitorView:
    """``EbMonitor`` as one ``fit_full_data`` sees it: the oracle's model k is the group's model ``models[k]``."""

    monitor: EbMonitor
    models: tuple[int, ...]

    def stage(self, name: str) -> None:
        self.monitor.stage(name)

    def observe(self, oracle: Any, hyperparameters: Sequence[Any], solving: Sequence[int]) -> None:
        self.monitor.observe(self.models, oracle, hyperparameters, solving)
