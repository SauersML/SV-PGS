"""Expected accuracy and power of the study's comparison, SNV/indel PGS against SNV/indel + SV PGS, per disease, from an
assumed genetic architecture and the study design: no genotypes and no fitting. A planning calculator; the simulation
grid (``benchmarks/power/grid.py``) checks it on real LD with the real pipeline.

Model. A disease is a liability threshold trait: liability = genetic value + noise, prevalence K, liability-scale
heritability h2. Of h2 a share ``sv_share`` is carried by causal SVs (``sv_share_of``: from the fraction of causal
variants that are SVs and their per-variant effect size relative to small variants). The SNV/indel arm captures the
small variants' part and the share ``tagged_by_small`` of the SVs' part through LD; the SV arm captures the SVs' part up to
max(tagged_by_small, sv_imputation_r2), because an imputed SV dosage is itself a function of the small variants around it
(its gain over them is how well one dosage represents a haplotype that a linear sum of small variants represents poorly,
not information the small variants lack; direct SV calls from reads are what add information).

Accuracy. Daetwyler et al. 2008: a PGS from N people captures r2 = N h2c / (N h2c + M) of the captured genetic value
h2c, with M the number of effects to estimate, here min(effective segments, causal variants) (a sparse prior estimates
only the causal ones well; Daetwyler et al. 2010). A disease is fitted on its 0/1 outcome, so h2c enters on the observed
scale (Lee et al. 2011, sample prevalence = K) and the result is converted back. AUC from liability R2 is Wray et al.
2010. Power is a paired Monte Carlo of the held-out comparison the report makes (DeLong delta AUC,
``held_out_comparison.paired_delta_auc``), conditional on the fitted models: training-set variability is not in it.

Every number here follows from the assumptions; the defaults are literature-scale values to edit, not measurements.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import stats

from sv_pgs.held_out_comparison import paired_delta_auc, power_weights
from benchmarks.power.uncertainty import rejection_summary


@dataclass(frozen=True)
class Architecture:
    """One disease's assumed genetic architecture and the study's numbers for it."""

    name: str
    prevalence: float
    heritability: float
    sv_share: float
    tagged_by_small: float
    sv_imputation_r2: float
    causal_variants: float
    effective_segments: float
    training_people: int
    test_people: int

    def __post_init__(self) -> None:
        if not 0.0 < self.prevalence < 1.0 or not 0.0 < self.heritability < 1.0:
            raise ValueError(f"{self.name}: prevalence and heritability lie in (0, 1)")
        for value in (self.sv_share, self.tagged_by_small, self.sv_imputation_r2):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{self.name}: shares and r2 lie in [0, 1]")
        if any(not math.isfinite(value) or value <= 0 for value in (self.causal_variants, self.effective_segments)):
            raise ValueError("causal variants and effective segments must be finite and positive")
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2
               for value in (self.training_people, self.test_people)):
            raise ValueError("training and test counts must be integers of at least two")


def sv_share_of(sv_causal_fraction: float, sv_effect_fold: float) -> float:
    """The share of h2 carried by SVs when that fraction of causal variants are SVs and an SV's effect SD is
    ``sv_effect_fold`` times a small variant's (equal per-SD variance otherwise): p f^2 / (p f^2 + 1 - p)."""
    if not 0 <= sv_causal_fraction <= 1 or not math.isfinite(sv_effect_fold) or sv_effect_fold <= 0:
        raise ValueError("causal fraction must be in [0, 1] and effect fold finite and positive")
    weighted = sv_causal_fraction * sv_effect_fold**2
    return weighted / (weighted + 1.0 - sv_causal_fraction)


def captured_heritability(architecture: Architecture) -> tuple[float, float]:
    """(SNV/indel arm, SV arm) liability-scale heritability each arm's variants can capture."""
    small = architecture.heritability * (1.0 - architecture.sv_share)
    structural = architecture.heritability * architecture.sv_share
    return (
        small + structural * architecture.tagged_by_small,
        small + structural * max(architecture.tagged_by_small, architecture.sv_imputation_r2),
    )


def observed_scale_factor(prevalence: float) -> float:
    """h2_observed / h2_liability for a 0/1 outcome with sample prevalence = K (Lee et al. 2011): z^2 / (K (1 - K))."""
    threshold = stats.norm.isf(prevalence)
    return stats.norm.pdf(threshold) ** 2 / (prevalence * (1.0 - prevalence))


def estimation_accuracy(captured: float, architecture: Architecture) -> float:
    """r2 of the PGS with the captured genetic value (Daetwyler et al. 2008) for the 0/1-outcome fit."""
    observed = captured * observed_scale_factor(architecture.prevalence)
    effects = min(architecture.effective_segments, architecture.causal_variants)
    return architecture.training_people * observed / (architecture.training_people * observed + effects)


def liability_r2(captured: float, architecture: Architecture) -> float:
    """The PGS's expected liability-scale R2: the captured heritability times its estimation accuracy."""
    return captured * estimation_accuracy(captured, architecture)


def auc_of(liability_variance: float, prevalence: float) -> float:
    """AUC of a predictor explaining ``liability_variance`` of the liability (Wray, Yang, Goddard and Visscher 2010)."""
    if liability_variance <= 0.0:
        return 0.5
    threshold = stats.norm.isf(prevalence)
    case_mean = stats.norm.pdf(threshold) / prevalence
    control_mean = -case_mean * prevalence / (1.0 - prevalence)
    spread = liability_variance * ((1.0 - liability_variance * case_mean * (case_mean - threshold)) + (1.0 - liability_variance * control_mean * (control_mean - threshold)))
    return float(stats.norm.cdf((case_mean - control_mean) * liability_variance / math.sqrt(spread)))


@dataclass(frozen=True)
class Expectation:
    small_auc: float
    sv_auc: float
    small_liability_r2: float
    sv_liability_r2: float
    power: float
    power_after_correction: float
    mean_delta_auc_simulated: float
    valid_replicates: int
    invalid_replicates: int
    power_mcse: float
    power_lower_95: float
    power_upper_95: float
    # The mean z of the paired test over the valid replicates: E[z], which the pre-registered panel weights follow.
    mean_z: float


def expectation(
    architecture: Architecture, replicates: int, seed: int, significance: float, tests: int, arm_error_correlation: float = 1.0,
) -> Expectation:
    """Analytic accuracies of both arms, and the power of the paired held-out DeLong test of their AUCs by a Monte Carlo
    of ``replicates`` test sets: one-sided at ``significance``, and Bonferroni-corrected over ``tests`` diseases.

    ``arm_error_correlation`` is the correlation of the two arms' small-variant estimation errors. The arms are separate
    fits: their small-variant effects come from the same people (most of each error is shared) but under different
    priors and columns, with their own approximation and draws. At 1 the error is shared exactly and cancels in the
    paired difference, so the power is an upper bound; below 1 each arm's error carries an unshared part, which the
    paired test must see through. A real study's value is measured from the two arms' fitted scores.
    """
    if not isinstance(replicates, int) or replicates <= 0 or not isinstance(tests, int) or tests <= 0 or not 0 < significance < 1:
        raise ValueError("replicates and tests must be positive integers, significance in (0, 1)")
    if not 0 <= arm_error_correlation <= 1:
        raise ValueError("the arms' error correlation must be in [0, 1]")
    small_captured, sv_captured = captured_heritability(architecture)
    small_r2, sv_r2 = estimation_accuracy(small_captured, architecture), estimation_accuracy(sv_captured, architecture)
    generator = np.random.default_rng(seed)
    threshold = stats.norm.isf(architecture.prevalence)
    extra = sv_captured - small_captured
    # The estimation noise of each arm, so corr^2(PGS, captured value) is its accuracy; the SV arm's noise extends the
    # SNV/indel arm's (the same small-variant effects, estimated from the same people).
    small_noise = small_captured * (1.0 - small_r2) / small_r2 if small_r2 > 0 else 0.0
    total_sv_noise = sv_captured * (1.0 - sv_r2) / sv_r2 if sv_r2 > 0 else 0.0
    sv_noise = max(total_sv_noise - small_noise, 0.0)
    critical = stats.norm.isf(significance)
    corrected = stats.norm.isf(significance / tests)
    z_scores, deltas = [], []
    for _replicate in range(replicates):
        size = architecture.test_people
        small_value = generator.normal(0.0, math.sqrt(small_captured), size)
        extra_value = generator.normal(0.0, math.sqrt(extra), size)
        remainder = generator.normal(0.0, math.sqrt(1.0 - sv_captured), size)
        cases = (small_value + extra_value + remainder > threshold).astype(np.float64)
        if cases.min() == cases.max():
            z_scores.append(math.nan)
            continue
        small_error = generator.normal(0.0, math.sqrt(small_noise), size)
        sv_small_error = small_error
        if arm_error_correlation < 1:
            unshared = generator.normal(0.0, math.sqrt(small_noise), size)
            sv_small_error = arm_error_correlation * small_error + math.sqrt(1.0 - arm_error_correlation**2) * unshared
        small_score = small_value + small_error
        sv_score = small_value + sv_small_error + extra_value + generator.normal(0.0, math.sqrt(sv_noise), size)
        comparison = paired_delta_auc(cases, small_score, sv_score)
        z_scores.append(comparison.z_score)
        deltas.append(comparison.estimate)
    z_values = np.asarray(z_scores)
    z_values = z_values[np.isfinite(z_values)]
    rate = rejection_summary(int(np.sum(z_values > critical)), len(z_values))
    return Expectation(
        small_auc=auc_of(liability_r2(small_captured, architecture), architecture.prevalence),
        sv_auc=auc_of(liability_r2(sv_captured, architecture), architecture.prevalence),
        small_liability_r2=liability_r2(small_captured, architecture),
        sv_liability_r2=liability_r2(sv_captured, architecture),
        power=rate['rate'],
        power_after_correction=float(np.mean(z_values > corrected)) if z_values.size else math.nan,
        mean_delta_auc_simulated=float(np.mean(deltas)) if deltas else math.nan,
        valid_replicates=int(z_values.size), invalid_replicates=replicates - int(z_values.size),
        power_mcse=rate['mcse'], power_lower_95=rate['lower'], power_upper_95=rate['upper'],
        mean_z=float(np.mean(z_values)) if z_values.size else math.nan,
    )


# Literature-scale assumptions for the 15-disease panel: (name, prevalence in an EHR biobank, liability-scale SNP
# heritability, the share of the cohort the disease is defined in). Edit them; nothing here is an All of Us statistic.
PANEL = (
    ("hypertension", 0.40, 0.20, 1.0), ("coronary_artery_disease", 0.08, 0.20, 1.0), ("asthma", 0.15, 0.15, 1.0),
    ("hyperlipidemia", 0.35, 0.15, 1.0), ("breast_cancer", 0.06, 0.13, 0.6), ("type2_diabetes", 0.20, 0.20, 1.0),
    ("atrial_fibrillation", 0.08, 0.15, 1.0), ("hypothyroidism", 0.12, 0.15, 1.0), ("copd", 0.08, 0.10, 1.0),
    ("depression", 0.20, 0.08, 1.0), ("gout", 0.05, 0.20, 1.0), ("cataract", 0.20, 0.10, 1.0),
    ("chronic_kidney_disease", 0.10, 0.08, 1.0), ("psoriasis", 0.03, 0.20, 1.0), ("prostate_cancer", 0.08, 0.25, 0.4),
)
"""(disease, prevalence, liability SNP-h2, fraction of the cohort eligible: breast cancer women, prostate cancer men)."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--people", type=int, default=100_000, help="genotyped cohort size")
    parser.add_argument("--folds", type=int, default=5, help="cross-validation folds (training share 1 - 1/folds)")
    parser.add_argument("--calibration-fraction", type=float, default=0.1, help="fraction of each development fold reserved from fitting")
    parser.add_argument("--sv-causal-fractions", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--sv-effect-folds", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--tagged-by-small", type=float, nargs="+", default=[0.3, 0.6])
    parser.add_argument("--sv-imputation-r2", type=float, default=0.8)
    parser.add_argument("--causal-variants", type=float, default=10_000)
    parser.add_argument("--effective-segments", type=float, default=60_000)
    parser.add_argument(
        "--arm-error-correlations", type=float, nargs="+", default=[1.0, 0.99, 0.95, 0.9],
        help="correlations of the two arms' small-variant estimation errors (1: shared exactly, the upper bound on power)",
    )
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--significance", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--panel-weights", action="store_true",
        help="write the run config's frozen panel block instead: weights proportional to each disease's E[z] at one common "
        "SV architecture (the first of each scenario list, the first arm error correlation), so they follow h2, n and "
        "prevalence and no SV biology (EVALUATION.md, the pre-registered panel)",
    )
    arguments = parser.parse_args()
    if arguments.folds < 2 or not 0 <= arguments.calibration_fraction < 1:
        parser.error("folds must be at least two and calibration-fraction in [0, 1)")
    rows = []
    expected_z = {}
    for name, prevalence, heritability, eligible in PANEL:
        people = int(arguments.people * eligible)
        base = Architecture(
            name=name, prevalence=prevalence, heritability=heritability, sv_share=0.0, tagged_by_small=0.0,
            sv_imputation_r2=arguments.sv_imputation_r2, causal_variants=arguments.causal_variants,
            effective_segments=arguments.effective_segments,
            training_people=int((people - people // arguments.folds) * (1 - arguments.calibration_fraction)),
            # A fixed-predictor planning calculation for ONE outer holdout, not a pooled CV power claim.
            test_people=people // arguments.folds,
        )
        if arguments.panel_weights:
            architecture = replace(
                base, sv_share=sv_share_of(arguments.sv_causal_fractions[0], arguments.sv_effect_folds[0]), tagged_by_small=arguments.tagged_by_small[0]
            )
            result = expectation(architecture, arguments.replicates, len(expected_z), arguments.significance, len(PANEL), arguments.arm_error_correlations[0])
            expected_z[name] = result.mean_z
            continue
        for sv_causal_fraction in arguments.sv_causal_fractions:
            for fold in arguments.sv_effect_folds:
                for tagged, correlation in itertools.product(arguments.tagged_by_small, arguments.arm_error_correlations):
                    architecture = replace(base, sv_share=sv_share_of(sv_causal_fraction, fold), tagged_by_small=tagged)
                    result = expectation(architecture, arguments.replicates, len(rows), arguments.significance, len(PANEL), correlation)
                    rows.append({
                        "disease": name, "prevalence": prevalence, "heritability": heritability, "people": people,
                        "cases": int(people * prevalence), "sv_causal_fraction": sv_causal_fraction, "sv_effect_fold": fold,
                        "sv_share_of_h2": round(architecture.sv_share, 4), "tagged_by_small": tagged,
                        "arm_error_correlation": correlation,
                        "snv_indel_auc": round(result.small_auc, 4), "with_sv_auc": round(result.sv_auc, 4),
                        "expected_delta_auc": round(result.sv_auc - result.small_auc, 5),
                        "simulated_delta_auc": round(result.mean_delta_auc_simulated, 5),
                        "power": round(result.power, 3), "power_bonferroni_15": round(result.power_after_correction, 3),
                        "valid_replicates": result.valid_replicates, "invalid_replicates": result.invalid_replicates,
                        "power_mcse": result.power_mcse, "power_lower_95": result.power_lower_95,
                        "power_upper_95": result.power_upper_95,
                        "training_people": architecture.training_people, "test_people": architecture.test_people,
                        "evaluation_scope": "one_outer_holdout_conditional_on_assumed_predictors",
                    })
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.panel_weights:
        weights = power_weights(np.array([expected_z[name] for name, *_rest in PANEL]))
        block = {"primary_fold": 0, "weights": {name: round(float(weight), 6) for (name, *_rest), weight in zip(PANEL, weights)}}
        arguments.output.write_text(json.dumps(block, indent=2) + "\n")
        print(f"wrote the panel block for {len(PANEL)} diseases to {arguments.output}")
        return
    with arguments.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {arguments.output}")


if __name__ == "__main__":
    main()
