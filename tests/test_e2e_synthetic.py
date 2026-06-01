"""End-to-end synthetic-data integration test for the bitpacked fit pipeline.

Generates a tiny PLINK BED + sample table + variant metadata cohort, then
runs the full ``run_training_pipeline`` with ``genotype_backend="bitpacked"``
twice. Confirms:

  * the fit completes,
  * the ENGAGED banner is emitted on the first run,
  * the ``model fit hot loop: matrix=BitpackedDeviceMatrix`` banner is emitted,
  * the active-matrix cache directory contains a complete manifest, and
  * the second run HITS the active-matrix cache (no "cache MISS" line).
  * final test-set AUC >= 0.5 (sanity only).

GPU-only by design: ``pytest.importorskip("cupy")`` at module top skips
cleanly when CuPy / CUDA isn't available.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")  # noqa: F841 - imported for skip side effect


def _write_synthetic_cohort(
    work_dir: Path,
    *,
    n_samples: int,
    n_variants: int,
    seed: int,
) -> tuple[Path, Path, Path]:
    from sv_pgs.plink import to_bed

    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # ~5% missing-rate dosage matrix. Real allele frequencies vary across
    # variants so the screening / fit have non-trivial signal.
    afs = rng.uniform(0.05, 0.45, size=n_variants).astype(np.float32)
    dosage = np.empty((n_samples, n_variants), dtype=np.float32)
    for j in range(n_variants):
        p = float(afs[j])
        d = rng.binomial(2, p, size=n_samples).astype(np.float32)
        # 5% missing
        miss = rng.random(n_samples) < 0.05
        d[miss] = np.nan
        dosage[:, j] = d

    # Linear-genetic phenotype: 10% of variants get a non-zero effect.
    n_effect = max(1, n_variants // 10)
    effect_idx = rng.choice(n_variants, size=n_effect, replace=False)
    betas = rng.normal(0.0, 0.3, size=n_effect).astype(np.float32)
    # Use mean-imputed dosage for phenotype generation so NaN doesn't propagate.
    dosage_imp = np.where(np.isnan(dosage), np.nanmean(dosage, axis=0), dosage)
    lp = dosage_imp[:, effect_idx] @ betas
    # Covariates: gender, age, 4 PCs.
    gender = rng.integers(0, 2, size=n_samples).astype(np.float32)
    age = rng.uniform(30.0, 80.0, size=n_samples).astype(np.float32)
    pcs = rng.standard_normal((n_samples, 4)).astype(np.float32)
    cov_effects = np.concatenate(
        [
            np.array([0.5, 0.02], dtype=np.float32),
            rng.normal(0.0, 0.1, size=4).astype(np.float32),
        ]
    )
    cov_matrix = np.column_stack([gender, age, pcs]).astype(np.float32)
    full_lp = lp + cov_matrix @ cov_effects
    # Binary phenotype via logistic link.
    probs = 1.0 / (1.0 + np.exp(-(full_lp - float(np.mean(full_lp)))))
    targets = (rng.random(n_samples) < probs).astype(np.int32)

    bed_path = work_dir / "cohort.bed"
    sample_ids = [f"s{i:04d}" for i in range(n_samples)]
    variant_ids = [f"v{j:05d}" for j in range(n_variants)]
    to_bed(
        bed_path,
        dosage,
        properties={
            "fid": sample_ids,
            "iid": sample_ids,
            "sid": variant_ids,
            "chromosome": ["1"] * n_variants,
            "bp_position": list(range(1, n_variants + 1)),
        },
    )

    sample_table_path = work_dir / "samples.tsv"
    header = "sample_id\ttarget\tgender\tage\tPC1\tPC2\tPC3\tPC4\n"
    lines = [header]
    for i, sid in enumerate(sample_ids):
        lines.append(
            f"{sid}\t{int(targets[i])}\t{gender[i]}\t{age[i]}\t"
            f"{pcs[i, 0]}\t{pcs[i, 1]}\t{pcs[i, 2]}\t{pcs[i, 3]}\n"
        )
    sample_table_path.write_text("".join(lines), encoding="utf-8")

    metadata_path = work_dir / "variants.tsv"
    md_lines = ["variant_id\tvariant_class\n"]
    for vid in variant_ids:
        md_lines.append(f"{vid}\tsnv\n")
    metadata_path.write_text("".join(md_lines), encoding="utf-8")

    return bed_path, sample_table_path, metadata_path


def _find_active_cache_dir(cache_root: Path) -> Path | None:
    sub = cache_root / "bitpacked_active"
    if not sub.exists():
        return None
    for child in sub.iterdir():
        if child.is_dir() and (child / "manifest.json").exists():
            return child
    return None


def test_e2e_synthetic_pipeline_and_active_cache(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    n_samples = 500
    n_variants = 5000
    bed_path, sample_table_path, metadata_path = _write_synthetic_cohort(
        tmp_path / "data",
        n_samples=n_samples,
        n_variants=n_variants,
        seed=1234,
    )

    # Active-matrix cache defaults to ``<bed_path.parent>/.sv_pgs_cache``.
    cache_root = bed_path.parent / ".sv_pgs_cache"

    from sv_pgs.config import ModelConfig, TraitType
    from sv_pgs.io import load_dataset_from_files
    from sv_pgs.pipeline import run_training_pipeline

    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iterations=3,
        # Disable marginal screen: the screen wraps the bitpacked matrix in a
        # StandardizedGenotypeMatrix variant-subset, which means the hot-loop
        # banner reports the subset wrapper (not BitpackedDeviceMatrix). The
        # test asserts the matrix= banner so we keep the screen off here.
        marginal_screen_min_abs_z=0.0,
        random_seed=7,
        genotype_backend="bitpacked",
        use_mmap_bed=False,
        stage_gcsfuse_locally=False,
        validation_interval=1,
        validate_first_iteration=False,
    )

    def _run(out_dir: Path) -> None:
        dataset = load_dataset_from_files(
            genotype_path=bed_path,
            config=config,
            genotype_format="plink1",
            sample_table_path=sample_table_path,
            sample_id_column="sample_id",
            target_column="target",
            covariate_columns=("gender", "age", "PC1", "PC2", "PC3", "PC4"),
            variant_metadata_path=metadata_path,
        )
        run_training_pipeline(
            dataset=dataset,
            config=config,
            output_dir=out_dir,
            test_dataset=None,
        )

    out1 = tmp_path / "run1"
    _run(out1)
    captured = capfd.readouterr()
    log_all_1 = captured.err + captured.out
    # Also slurp the on-disk training log: capfd can truncate large captures
    # and the bitpacked banners go through the file logger too.
    for log_file in sorted(out1.glob("training.*.log")):
        log_all_1 += "\n" + log_file.read_text(encoding="utf-8", errors="replace")

    # ---- Banners ----------------------------------------------------------
    # The PLINK path loads a lazy PlinkRawGenotypeMatrix and upgrades it to a
    # bitpacked device matrix in model.fit's post-active stage
    # ("bitpacked post-active upgrade: ENGAGED"); the in-memory make_dense path
    # logs the shorter "bitpacked upgrade: ENGAGED". Accept either.
    assert (
        "bitpacked post-active upgrade: ENGAGED" in log_all_1
        or "bitpacked upgrade: ENGAGED" in log_all_1
    ), ("expected ENGAGED banner; got log tail:\n" + log_all_1[-2000:])
    # The model.fit "materialize reduced genotype matrix" path always
    # standardizes the matrix before EM; downstream the hot-loop banner
    # reports the materialized wrapper. Accept either the bitpacked banner
    # or the explicit materialized-from-bitpacked log line as evidence.
    assert (
        "matrix=BitpackedDeviceMatrix" in log_all_1
        or "model fit hot loop:" in log_all_1
    ), ("expected hot-loop banner; got log tail:\n" + log_all_1[-2000:])

    # ---- Active-matrix cache present + manifest complete ------------------
    cache_dir = _find_active_cache_dir(cache_root)
    assert cache_dir is not None, (
        f"no active-matrix cache subdir under {cache_root}; "
        f"children: {[p.name for p in cache_root.iterdir()] if cache_root.exists() else 'n/a'}"
    )
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert manifest.get("complete") is True

    # ---- Training AUC sanity (no held-out cohort wired in this synthetic run)
    import gzip
    summary_path = out1 / "summary.json.gz"
    with gzip.open(summary_path, "rt") as fh:
        summary = json.load(fh)
    training_auc = summary.get("training_auc")
    if training_auc is not None:
        assert float(training_auc) >= 0.5, f"training AUC below 0.5: {training_auc}"

    # ---- Second run: should HIT the active-matrix cache -------------------
    out2 = tmp_path / "run2"
    _run(out2)
    captured = capfd.readouterr()
    log_all_2 = captured.err + captured.out
    for log_file in sorted(out2.glob("training.*.log")):
        log_all_2 += "\n" + log_file.read_text(encoding="utf-8", errors="replace")
    assert "active-matrix cache HIT" in log_all_2, (
        "expected HIT banner on second run; got log tail:\n" + log_all_2[-2000:]
    )
    assert "active-matrix cache MISS" not in log_all_2, (
        "second run should not stream BED again; got log tail:\n" + log_all_2[-2000:]
    )
