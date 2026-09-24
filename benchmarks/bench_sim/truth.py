"""bench-sim step 4: scenario truths drawn from the pre-registered family (PREREG.md section 3).

scenario(seed) draws every parameter from its own seed, computes the true genetic value from truth
genotypes, and returns the phenotype. Dev seeds 0-23 are public; test seeds derive from the sealed
master seed and are written only under sealed/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SHAPES = ("gaussian", "student_t", "laplace", "variance_grid", "two_point", "sparse_mixture", "lognormal_scale", "directional")
MODES = ("none", "scale", "probability", "both")
SUPERPOP_AFR, SUPERPOP_EUR = 0, 3
SNV, INDEL, TR, SV = 0, 1, 2, 3


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def unit_variance_draws(rng: np.random.Generator, shape: str, count: int, params: dict) -> np.ndarray:
    """Standardized effect draws (variance 1) of one shape family, before the per-variant scale."""
    if shape == "gaussian":
        return rng.standard_normal(count)
    if shape == "student_t":
        nu = params["nu"]
        return rng.standard_t(nu, size=count) * np.sqrt((nu - 2.0) / nu)
    if shape in ("laplace", "directional"):
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=count)
    if shape == "variance_grid":
        weights = np.asarray(params["grid_weights"])
        variances = np.array([1.0, 10.0, 100.0])
        component = rng.choice(3, size=count, p=weights)
        return rng.standard_normal(count) * np.sqrt(variances[component] / float(weights @ variances))
    if shape == "two_point":
        return rng.choice(np.array([-1.0, 1.0]), size=count)
    if shape == "sparse_mixture":
        large = rng.random(count) < 0.01
        return rng.standard_normal(count) * np.where(large, 10.0, 1.0) / np.sqrt(0.99 + 0.01 * 100.0)
    if shape == "lognormal_scale":
        tau = params["tau"]
        return rng.standard_normal(count) * np.exp(0.5 * rng.normal(0.0, tau, size=count)) / np.exp(tau * tau / 4.0)
    raise ValueError(shape)


def random_spline(rng: np.random.Generator, values: np.ndarray, sd: float) -> np.ndarray:
    """A random natural cubic spline of a continuous annotation: 4 knots at data quantiles, centred."""
    finite = values[np.isfinite(values)]
    knots = np.quantile(finite, [0.05, 0.35, 0.65, 0.95])
    clipped = np.clip(values, knots[0], knots[-1])

    def truncated_cube(point: np.ndarray, knot: float) -> np.ndarray:
        return np.maximum(point - knot, 0.0) ** 3

    span = knots[-1] - knots[-2]
    basis = [clipped]
    for knot in knots[:-2]:
        basis.append((truncated_cube(clipped, knot) - truncated_cube(clipped, knots[-1])) / (knots[-1] - knot)
                     - (truncated_cube(clipped, knots[-2]) - truncated_cube(clipped, knots[-1])) / span)
    basis_matrix = np.column_stack(basis)
    spread = basis_matrix.std(axis=0)
    basis_matrix = (basis_matrix[:, spread > 0] - basis_matrix[:, spread > 0].mean(axis=0)) / spread[spread > 0]
    curve = basis_matrix @ rng.normal(0.0, sd, size=basis_matrix.shape[1])
    return curve - curve.mean()


def draw_parameters(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    params: dict = {"seed": seed}
    params["h2"] = log_uniform(rng, 0.01, 0.2)
    params["pi"] = log_uniform(rng, 1e-4, 3e-2)
    for name in ("sv", "tr"):
        params[f"{name}_mode"] = str(rng.choice(MODES))
        params[f"{name}_scale_fold"] = log_uniform(rng, 1.5, 5.0)
        params[f"{name}_probability_fold"] = log_uniform(rng, 2.0, 30.0)

    def shape_params(shape: str) -> dict:
        if shape == "student_t":
            return {"nu": float(rng.uniform(2.2, 8.0))}
        if shape == "variance_grid":
            return {"grid_weights": rng.dirichlet(np.ones(3)).tolist()}
        if shape == "lognormal_scale":
            return {"tau": float(rng.uniform(0.5, 2.0))}
        if shape == "directional":
            return {"direction_probability": float(rng.uniform(0.5, 0.9))}
        return {}

    params["shape"] = str(rng.choice(SHAPES))
    params["shape_params"] = shape_params(params["shape"])
    params["sv_own_shape"] = bool(rng.random() < 0.5)
    params["sv_shape"] = str(rng.choice(SHAPES)) if params["sv_own_shape"] else params["shape"]
    params["sv_shape_params"] = shape_params(params["sv_shape"]) if params["sv_own_shape"] else params["shape_params"]
    params["alpha_snv"] = float(rng.uniform(-1.0, 0.0))
    params["alpha_sv"] = params["alpha_snv"] + (float(rng.uniform(-0.5, 0.5)) if rng.random() < 0.5 else 0.0)
    params["plateau"] = log_uniform(rng, 1e-3, 2e-2) if rng.random() < 0.5 else 0.0
    params["frequency_source"] = str(rng.choice(("cohort", "afr", "eur")))
    params["ld_gamma"] = float(rng.uniform(-0.5, 0.0)) if rng.random() < 0.5 else 0.0
    params["annotation_mode"] = str(rng.choice(MODES))
    params["annotation_seed"] = int(rng.integers(2**31))
    params["dominance"] = bool(rng.random() < 0.3)
    params["tr_saturating"] = bool(rng.random() < 0.3)
    params["link_c"] = float(rng.uniform(0.0, 0.15)) if rng.random() < 0.2 else 0.0
    params["hetero_kappa"] = float(rng.uniform(0.0, 0.5))
    params["t_noise"] = bool(rng.random() < 0.3)
    params["beta_sex"] = float(rng.normal(0.0, 0.3))
    params["beta_age"] = float(rng.normal(0.0, 0.2))
    params["beta_batch"] = float(rng.normal(0.0, 0.1))
    params["ancestry_means"] = rng.normal(0.0, 0.3, size=5).tolist()
    params["binary"] = bool(rng.random() >= 0.6)
    params["prevalence"] = log_uniform(rng, 0.01, 0.3)
    params["effect_seed"] = int(rng.integers(2**31))
    return params


def genetic_value(params: dict, root: Path) -> tuple[np.ndarray, dict]:
    variants = np.load(root / "variants.npz")
    annotations = np.load(root / "annotations.npz")
    truth = np.load(root / "truth_G.npy", mmap_mode="r")
    cls = variants["cls"]
    n_var, size = truth.shape
    rng = np.random.default_rng(params["effect_seed"])
    annotation_rng = np.random.default_rng(params["annotation_seed"])
    structural = cls >= TR

    # Causal probability.
    logit_pi = np.full(n_var, np.log(params["pi"] / (1.0 - params["pi"])))
    for name, members in (("sv", cls == SV), ("tr", cls == TR)):
        if params[f"{name}_mode"] in ("probability", "both"):
            logit_pi[members] += np.log(params[f"{name}_probability_fold"])
    log_scale = np.zeros(n_var)
    for name, members in (("sv", cls == SV), ("tr", cls == TR)):
        if params[f"{name}_mode"] in ("scale", "both"):
            log_scale[members] += 2.0 * np.log(params[f"{name}_scale_fold"])
    binary_annotations = {"in_gene": annotations["in_gene"], "in_exon": annotations["in_exon"], "in_repeat": annotations["in_repeat"]}
    continuous_annotations = {"log_tss_distance": annotations["log_tss_distance"], "log_sv_length": annotations["log_sv_length"]}
    if params["annotation_mode"] in ("scale", "both"):
        for values in binary_annotations.values():
            log_scale += annotation_rng.normal(0.0, 0.5) * values
        for values in continuous_annotations.values():
            log_scale += random_spline(annotation_rng, values, 0.4)
    if params["annotation_mode"] in ("probability", "both"):
        for values in binary_annotations.values():
            logit_pi += annotation_rng.normal(0.0, 0.7) * values
        for values in continuous_annotations.values():
            logit_pi += random_spline(annotation_rng, values, 0.4)
    causal = np.flatnonzero(rng.random(n_var) < 1.0 / (1.0 + np.exp(-logit_pi)))

    # Frequency and LD dependence of the per-SD variance.
    genotypes = np.asarray(truth[causal], dtype=np.float64)
    cohort_p = genotypes.mean(axis=1) / 2.0
    frequency = {
        "cohort": cohort_p,
        "afr": variants["superpop_af"][SUPERPOP_AFR][causal].astype(np.float64),
        "eur": variants["superpop_af"][SUPERPOP_EUR][causal].astype(np.float64),
    }[params["frequency_source"]]
    floor = max(params["plateau"], 1.0 / float(variants["donor_an"]))
    frequency = np.clip(frequency, floor, 1.0 - floor)
    alpha = np.where(structural[causal], params["alpha_sv"], params["alpha_snv"])
    log_scale_causal = log_scale[causal] + (1.0 + alpha) * np.log(2.0 * frequency * (1.0 - frequency))
    if params["ld_gamma"]:
        log_scale_causal += params["ld_gamma"] * np.log(annotations["ld_score"][causal])

    # Standardized effects of each class's shape family.
    draws = np.empty(causal.size)
    sv_rows = structural[causal]
    draws[~sv_rows] = unit_variance_draws(rng, params["shape"], int((~sv_rows).sum()), params["shape_params"])
    draws[sv_rows] = unit_variance_draws(rng, params["sv_shape"], int(sv_rows.sum()), params["sv_shape_params"])
    for shape_key, params_key, rows in (("shape", "shape_params", ~sv_rows), ("sv_shape", "sv_shape_params", sv_rows)):
        if params[shape_key] == "directional":
            magnitude = np.abs(draws[rows])
            direction = np.where(
                (cls[causal][rows] == SV) & (variants["len_change"][causal][rows] < 0) & annotations["in_exon"][causal][rows],
                -1.0, rng.choice(np.array([-1.0, 1.0])),
            )
            aligned = rng.random(int(rows.sum())) < params[params_key]["direction_probability"]
            draws[rows] = magnitude * np.where(aligned, direction, -direction)
    standardized_effect = draws * np.exp(0.5 * log_scale_causal)
    sd = genotypes.std(axis=1)
    per_allele = np.divide(standardized_effect, sd, out=np.zeros_like(sd), where=sd > 0)
    centred = genotypes - genotypes.mean(axis=1, keepdims=True)
    additive_parts = per_allele[:, None] * centred

    # Non-additivity.
    total = additive_parts.sum(axis=0)
    dominance_rows = np.zeros(causal.size, dtype=bool)
    if params["dominance"]:
        dominance_rows = (cls[causal] == SV) & annotations["in_exon"][causal]
        if dominance_rows.any():
            ratio = rng.uniform(-0.5, 1.0, size=int(dominance_rows.sum()))
            heterozygous = (genotypes[dominance_rows] == 1.0).astype(np.float64)
            deviation = (ratio * per_allele[dominance_rows])[:, None] * (heterozygous - heterozygous.mean(axis=1, keepdims=True))
            total = total + deviation.sum(axis=0)
    saturating_loci = 0
    if params["tr_saturating"]:
        locus = annotations["repeat_locus"][causal]
        tr_rows = (cls[causal] == TR) & (locus >= 0)
        for locus_id in np.unique(locus[tr_rows]):
            members = np.flatnonzero(tr_rows & (locus == locus_id))
            length = (variants["len_change"][causal][members][:, None] * genotypes[members]).sum(axis=0)
            carriers = np.abs(length[length != 0])
            if carriers.size == 0:
                continue
            scale0 = float(np.median(carriers))
            curve = np.tanh(length / scale0)
            curve -= curve.mean()
            additive = additive_parts[members].sum(axis=0)
            if curve.std() == 0 or additive.std() == 0:
                continue
            magnitude = additive.std() / curve.std()
            total = total - additive + magnitude * curve * np.sign(np.corrcoef(additive, curve)[0, 1])
            saturating_loci += 1
    value = total
    if params["link_c"]:
        standardized_value = (value - value.mean()) / value.std()
        value = value + params["link_c"] * value.std() * (standardized_value ** 2 - 1.0)
    value = (value - value.mean()) / value.std() * np.sqrt(params["h2"])

    structural_share = 0.0
    if value.std() > 0:
        structural_part = additive_parts[sv_rows].sum(axis=0) if sv_rows.any() else np.zeros(size)
        structural_share = float(np.cov(structural_part, total)[0, 1] / np.var(total, ddof=1))
    truth_summary = {
        "causal_count": int(causal.size),
        "causal_by_class": np.bincount(cls[causal], minlength=4).tolist(),
        "structural_share": structural_share,
        "dominance_count": int(dominance_rows.sum()),
        "saturating_loci": saturating_loci,
    }
    scale_factor = np.sqrt(params["h2"]) / total.std() if total.std() > 0 else 0.0
    effects = {"causal": causal, "per_allele": per_allele * scale_factor, "genotype_means": genotypes.mean(axis=1)}
    return value, {"summary": truth_summary, "effects": effects}


def phenotype(params: dict, value: np.ndarray, root: Path) -> np.ndarray:
    samples = np.load(root / "samples.npz")
    rng = np.random.default_rng(params["effect_seed"] + 1)
    age = samples["age"]
    age_standardized = (age - age.mean()) / age.std()
    weight = np.exp(params["hetero_kappa"] * age_standardized)
    weight /= weight.mean()
    noise = rng.standard_t(5, size=value.size) / np.sqrt(5.0 / 3.0) if params["t_noise"] else rng.standard_normal(value.size)
    residual = noise * np.sqrt((1.0 - params["h2"]) * weight)
    ancestry = samples["realized_proportions"] @ np.asarray(params["ancestry_means"])
    liability = (value + residual + params["beta_sex"] * samples["sex"] + params["beta_age"] * age_standardized
                 + params["beta_batch"] * samples["batch"] + ancestry - ancestry.mean())
    if params["binary"]:
        threshold = np.quantile(liability, 1.0 - params["prevalence"])
        return (liability > threshold).astype(np.float64)
    return liability


def build(seed: int, root: Path, out: Path) -> dict:
    return write_scenario(draw_parameters(seed), root, out)


def write_scenario(params: dict, root: Path, out: Path) -> dict:
    """The scenario of ``params`` on the cohort chromosome ``root``: truth.npz and scenario.json under ``out``."""
    value, truth = genetic_value(params, root)
    observed_phenotype = phenotype(params, value, root)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "truth.npz", genetic_value=value, phenotype=observed_phenotype, **truth["effects"])
    record = {"params": params, "summary": truth["summary"]}
    (out / "scenario.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    return record


def sealed_seeds(master_hex: str, count: int) -> list[int]:
    return [int.from_bytes(hashlib.sha256(f"{master_hex}:{index}".encode()).digest()[:4], "little") for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="cohort chromosome directory")
    parser.add_argument("--out", required=True, help="bench-sim root; writes dev/ and sealed/")
    parser.add_argument("--set", choices=("dev", "test"), required=True)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    root, out = Path(args.dir), Path(args.out)
    if args.set == "dev":
        for seed in range(args.count or 24):
            record = build(seed, root, out / "dev" / f"scenario_{seed:03d}")
            print(seed, json.dumps(record["summary"]), flush=True)
    else:
        master = (out / "sealed" / "master_seed.txt").read_text().strip()
        for index, seed in enumerate(sealed_seeds(master, args.count or 64)):
            record = build(seed, root, out / "sealed" / f"scenario_{index:03d}")
            print(index, "built", flush=True)


if __name__ == "__main__":
    main()
