"""Imputation reliability of a stored genotype column, and the unit contract that carries it to the prior.

r^2 = corr^2(D, G) between a stored column D and the true genotype G is fixed from truth outside the fit (or taken
from the imputation's own report). How it reaches the prior depends on what D is, which ``ColumnMeasurement`` states
per record: the prior is on raw effects (per ALT allele, per copy), and its standardized form has variance
kappa^2 Var(D) tau^2 = r^2 Var(G) tau^2 with kappa = Cov(G, D) / Var(D) the calibration slope. For a calibrated
conditional mean (kappa = 1) the reliability is already in Var(D) and enters only the true genotype's variance
Var(G) = Var(D) / r^2, the frequency function's argument; a prior offset of log r^2 on top of the per-unit
log Var(D) counted it twice (review F21). r^2, a squared correlation, is invariant to affine maps of D, such as
the codec's scale and centring; a non-affine map changes it. The piece here:

- ``ColumnMeasurement``: raw unit, encoding scale, calibration slope and
  reliability with its source, from store metadata to the prior's offset and
  frequency argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sv_pgs._typing import F64Array, NDArray


def checked_log_reliability(values: NDArray, source: str) -> F64Array:
    """The records' log reliabilities log r^2, against the one contract they all meet.

    Every source of a record's reliability ends here: this module's fitted model,
    ``measurement_model.log_reliability_offsets`` on truth pairs, a store's reported
    imputation r^2, and a caller's explicit offsets. The contract is r^2 in [0, 1],
    so the value is at most 0, and -inf is the record whose stored column carries
    no information about its genotype: its standardized prior variance
    r^2 Var(G) tau^2 is 0, its effect is exactly zero, and the fit leaves it out.

    Anything else is corrupt metadata rather than a measurement, and raises here
    instead of reaching the prior. A value above 0 is no squared correlation; a nan would be dropped
    by candidate selection, which tests offsets for finiteness, so a record with
    unreadable metadata would pass for one that was measured and found empty.
    """
    offsets = np.asarray(values, dtype=np.float64)
    invalid = ~(offsets <= 0.0)
    if invalid.any():
        first = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"{source}: every record's log reliability must be <= 0, a log r^2 with r^2 in [0, 1]; "
            f"record {first} has {offsets[first]} ({int(invalid.sum())} of {offsets.size} records)"
        )
    return offsets


@dataclass(frozen=True)
class ColumnMeasurement:
    """The unit and measurement contract of the prior's columns: what one stored column D_j measures, in which unit.

    The prior is stated in raw units first. The effect b_j of one unit of the true genotype G_j (one ALT allele of a
    dosage record, one copy of a copy-number record: ``raw_unit``) has prior variance tau_j^2 = e^(t + a_j), a draw t of
    the class's mixing density and a_j the learned log-variance terms (the frequency function h(log Var G_j), the
    annotation smooths). The fit sees the column standardized, x_j = (D_j - mean) / sd(D_j), with D_j = code /
    ``codes_per_unit`` in raw units. Where the conditional mean of G_j given D_j is linear with slope
    kappa_j = Cov(G_j, D_j) / Var(D_j) (``calibration_slope``), E[y | D] carries kappa_j b_j per unit of D_j, so the
    coefficient on x_j is beta_j = kappa_j sd(D_j) b_j and

        Var(beta_j) = kappa_j^2 Var(D_j) tau_j^2 = Corr(G_j, D_j)^2 Var(G_j) tau_j^2

    (the two forms agree through r_j^2 = Corr(G, D)^2 = kappa_j^2 Var(D_j) / Var(G_j)). So the prior's offset, with
    coefficient exactly 1, is log(kappa_j^2 Var(D_j)), and the true genotype's variance, the argument of the frequency
    function, is Var(G_j) = kappa_j^2 Var(D_j) / r_j^2.

    A calibrated conditional mean, D = E[G | O] for the evidence O (a GLIMPSE2 or Beagle DS, the measurement step's
    recalibrated D*, a hard call measured exactly), has Cov(G, D) = Var(D), so kappa = 1: its reduced variance already
    carries the lost information, r^2 = Var(D) / Var(G), and multiplying the prior by r^2 again would shrink a poorly
    measured column twice (Var(beta) = r^2 Var(D) tau^2 = r^4 Var(G) tau^2; review F21). The reliability instead enters
    through Var(G), the frequency function's argument. A deliberately uncalibrated column (a confident draw, whose
    variance stays near Var(G)) has kappa^2 = r^2 Var(G) / Var(D) < 1 and takes it here, measured from truth pairs
    (``measurement_model.fit_measurement_model``'s scales), never guessed from its r^2.

    Measured on bench-real's Beagle-imputed SV overlay (chr16, 637 varying records, the panel's calls as truth [real
    genotypes]): the energy-weighted kappa of the stored DS is 0.87 (median 0.81), and among records with true
    r^2 > 0.1 the target log(Corr^2 Var G) regressed on log Var(D) and log DR2 has coefficients 0.97 and -1.55: the
    calibrated-mean offset log Var(D) tracks it (RMS error 1.07 in log variance), the doubly counted log DR2 + log Var(D)
    does not (1.44). Var(D) / DR2 estimates Var(G) with median log error -0.03 (RMS 1.21).

    ``log_reliability`` is log r_j^2 (``checked_log_reliability``: at most 0, -inf where the column carries no
    information), and ``reliability_source`` says where it came from (a store's reported imputation r^2, truth pairs,
    or none: measured exactly), recorded with the fit.
    """

    raw_unit: NDArray
    codes_per_unit: F64Array
    calibration_slope: F64Array
    log_reliability: F64Array
    reliability_source: str

    @classmethod
    def build(
        cls, record_count: int, *, log_reliability: NDArray | None = None, codes_per_unit: NDArray | None = None,
        calibration_slope: NDArray | None = None, raw_unit: NDArray | None = None, reliability_source: str | None = None,
    ) -> ColumnMeasurement:
        """The contract of ``record_count`` stored columns. Absent inputs are the store's defaults and say so: no
        reliability is measurement without error (r^2 = 1), no encoding scale is the store's 127 codes per ALT
        allele, no calibration slope is a calibrated conditional mean (kappa = 1)."""
        from sv_pgs.dosage_store import CODES_PER_DOSAGE  # noqa: PLC0415 - the store imports nothing from here

        def per_record(values: NDArray | None, default: float, name: str) -> F64Array:
            array = np.full(record_count, default) if values is None else np.asarray(values, dtype=np.float64)
            if array.shape != (record_count,):
                raise ValueError(f"{name} needs one value per record ({record_count}), got shape {array.shape}")
            return array

        reliability = per_record(log_reliability, 0.0, "log_reliability")
        checked_log_reliability(reliability, reliability_source or "the column measurement's log reliability")
        units = per_record(codes_per_unit, float(CODES_PER_DOSAGE), "codes_per_unit")
        slope = per_record(calibration_slope, 1.0, "calibration_slope")
        if not (np.all(units > 0.0) and np.all(np.isfinite(units))):
            raise ValueError("every record needs a positive, finite encoding scale (codes per raw unit)")
        if not (np.all(slope > 0.0) and np.all(np.isfinite(slope))):
            raise ValueError("every calibration slope must be positive and finite: a column whose conditional mean does not rise with it is not a measurement of its genotype")
        labels = np.full(record_count, "ALT allele", dtype=object) if raw_unit is None else np.asarray(raw_unit, dtype=object)
        if labels.shape != (record_count,):
            raise ValueError("raw_unit needs one label per record")
        source = reliability_source or ("none: every record measured exactly" if log_reliability is None else "the caller's log reliability")
        return cls(raw_unit=labels, codes_per_unit=units, calibration_slope=slope, log_reliability=reliability, reliability_source=source)

    def standardized_terms(self, members: NDArray, code_scales: NDArray) -> tuple[F64Array, F64Array]:
        """(offset, log genotype variance) of the prior's ``members`` (record indices) from their training code SDs.

        The offset is log(kappa_j^2 Var(D_j)) less its largest over the members: a constant shift of every member's
        log prior variance, which the class densities' common location carries, so every offset stays at or below 0.
        The log genotype variance log Var(G_j) = log(kappa_j^2 Var(D_j)) - log r_j^2 is the frequency function's
        argument (``annotation_design.frequency_annotation``)."""
        rows = np.asarray(members, dtype=np.int64)
        scales = np.asarray(code_scales, dtype=np.float64) / self.codes_per_unit[rows]
        if scales.shape != rows.shape or not np.all(scales > 0.0):
            raise ValueError("every member needs a positive training spread")
        log_signal = 2.0 * (np.log(self.calibration_slope[rows]) + np.log(scales))
        offset = log_signal - float(np.max(log_signal)) if rows.size else log_signal
        return offset, log_signal - self.log_reliability[rows]
