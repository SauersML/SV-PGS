import numpy as np
from sklearn.metrics import roc_auc_score

from sv_pgs import BayesianPGS, BenchmarkConfig, ModelConfig, TraitType, VariantClass, VariantRecord, run_benchmark_suite


def _synthetic_binary_data():
    rng = np.random.default_rng(7)
    n = 160
    covariates = rng.normal(size=(n, 2)).astype(np.float32)
    x0 = rng.normal(size=n).astype(np.float32)
    x1 = x0.copy()
    x2 = (x0 + rng.normal(scale=0.05, size=n)).astype(np.float32)
    x3 = rng.normal(size=n).astype(np.float32)
    x4 = rng.normal(size=n).astype(np.float32)
    genotypes = np.column_stack([x0, x1, x2, x3, x4]).astype(np.float32)
    genotypes[rng.choice(n, size=12, replace=False), 4] = np.nan
    logits = 1.4 * x0 - 1.0 * x4 + 0.6 * covariates[:, 0] - 0.4 * covariates[:, 1]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    targets = rng.binomial(1, probabilities).astype(np.float32)
    records = [
        VariantRecord("v0", VariantClass.SNV, "na", "1", 100),
        VariantRecord("v1", VariantClass.SNV, "na", "1", 101),
        VariantRecord("v2", VariantClass.DELETION_SHORT, "short", "1", 105, cluster_id="c1"),
        VariantRecord("v3", VariantClass.SNV, "na", "1", 2000000),
        VariantRecord("v4", VariantClass.DUPLICATION_SHORT, "short", "1", 110, cluster_id="c1"),
    ]
    return genotypes, covariates, targets, records


def test_binary_fit_graph_ties_and_roundtrip(tmp_path):
    genotypes, covariates, targets, records = _synthetic_binary_data()
    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iters=12,
        max_inner_pcg_iters=80,
        correlation_threshold=0.95,
        tile_size=8,
    )
    model = BayesianPGS(config).fit(genotypes, covariates, targets, records)

    state = model.state
    assert state is not None
    assert state.tie_map.kept_indices.tolist() == [0, 2, 3, 4]
    assert state.graph.src.shape[0] >= 1
    probabilities = model.predict_proba(genotypes, covariates)[:, 1]
    assert roc_auc_score(targets, probabilities) > 0.75

    export_path = tmp_path / "artifact"
    model.export(export_path)
    loaded = BayesianPGS.load(export_path)
    np.testing.assert_allclose(
        loaded.decision_function(genotypes, covariates),
        model.decision_function(genotypes, covariates),
        atol=1e-5,
    )


def test_benchmark_suite_runs_same_code_path():
    genotypes, covariates, targets, records = _synthetic_binary_data()
    split = 120
    metrics = run_benchmark_suite(
        train_genotypes=genotypes[:split],
        train_covariates=covariates[:split],
        train_targets=targets[:split],
        test_genotypes=genotypes[split:],
        test_covariates=covariates[split:],
        test_targets=targets[split:],
        records=records,
        benchmark_config=BenchmarkConfig(
            shared_config=ModelConfig(
                trait_type=TraitType.BINARY,
                max_outer_iters=8,
                max_inner_pcg_iters=60,
                correlation_threshold=0.95,
                tile_size=8,
            )
        ),
    )

    assert set(metrics) == {"current_snv_score", "snv_only_bayesr", "joint_snv_sv_bayesr"}
    assert metrics["joint_snv_sv_bayesr"].auc is not None
    assert metrics["joint_snv_sv_bayesr"].log_loss is not None
