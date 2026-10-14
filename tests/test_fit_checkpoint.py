"""A preempted job must preserve completed folds without changing their inference."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from sv_pgs import stage2_wiring
from sv_pgs.config import TraitType
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.fit_checkpoint import checkpoint_timings, load_checkpoint, save_checkpoint
from sv_pgs.scale_mixture_ep import MixtureHyperparameters
from tests.test_draw_laws import _law
from tests.test_fit_model import _budget, _certificate


def _fit(seed, mixture=False):
    rng = np.random.default_rng(seed)
    law = _law(rng, rows=3, draws=8) if mixture else rng.normal(size=(3, 8))
    scoring = ScoringModel(
        np.arange(3), np.zeros(3), np.ones(3), rng.normal(size=3), law,
        np.zeros(1), TraitType.BINARY, 0.3, np.zeros((1, 8)), np.zeros((1, 1)), False,
    )
    return stage2_wiring._ModelFit(
        scoring, 1.0, MixtureHyperparameters(np.zeros(2), np.zeros(1)), _certificate(1, rng), "prior",
    )


def _equal(left, right):
    for field in dataclasses.fields(left.scoring):
        if field.name != "posterior_draws":
            np.testing.assert_array_equal(getattr(left.scoring, field.name), getattr(right.scoring, field.name))
    np.testing.assert_array_equal(left.scoring.posterior_draws.tile(0, 3), right.scoring.posterior_draws.tile(0, 3))
    assert type(right.scoring.trait_type) is TraitType
    for field in dataclasses.fields(left.certificate):
        a, b = getattr(left.certificate, field.name), getattr(right.certificate, field.name)
        if isinstance(a, tuple):
            assert isinstance(b, tuple) and len(a) == len(b)
            for first, second in zip(a, b):
                np.testing.assert_array_equal(first, second)
        else:
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("mixture", [False, True])
def test_checkpoint_preserves_certificates_and_draw_laws(tmp_path, mixture):
    fit = _fit(42, mixture)
    path = tmp_path / "checkpoints" / "key" / "model0.npz"
    save_checkpoint(path, fit, "key", 12.5)
    loaded = load_checkpoint(path, "key")
    _equal(fit, loaded)
    assert type(loaded.scoring.posterior_draws) is type(fit.scoring.posterior_draws)
    assert checkpoint_timings(tmp_path) == [{"model_index": 0, "elapsed_seconds": 12.5, "key": "key"}]
    with pytest.raises(ValueError, match="different fitting problem"):
        load_checkpoint(path, "other")


def test_preemption_resumes_completed_folds_with_original_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(stage2_wiring, "store_measurement", lambda *args: None)
    calls = []
    def engine(*args):
        seed = args[7]
        calls.append(seed)
        if len(calls) == 2:
            (args[6] / "partial-scratch").write_text("incomplete")
            raise RuntimeError("preempted")
        # Each binary model is a group of its own (``stage2_wiring.model_groups``), fitted alone.
        return [_fit(seed)]
    monkeypatch.setattr(stage2_wiring, "_fit_group", engine)
    arguments = dict(
        store=SimpleNamespace(), store_columns=np.arange(6), covariates=np.ones((6, 1)),
        covariate_columns=np.ones((3, 1), dtype=bool), targets=np.zeros((6, 3)),
        training=np.ones((6, 3), dtype=bool), trait_types=(TraitType.BINARY,) * 3,
        log_variance_offset=None, budget=_budget(), work_dir=tmp_path, seed=91,
        draw_count=8, checkpoint_key="problem",
    )
    with pytest.raises(RuntimeError, match="preempted"):
        stage2_wiring.fit_models(**arguments)
    resumed = stage2_wiring.fit_models(**arguments)
    seeds = [stage2_wiring._seed(91, index) for index in range(3)]
    assert calls == [seeds[0], seeds[1], seeds[1], seeds[2]]
    assert not (tmp_path / "model1").exists()
    for index, scoring in enumerate(resumed.scoring):
        np.testing.assert_array_equal(scoring.coefficients, _fit(seeds[index]).scoring.coefficients)
    monkeypatch.setattr(stage2_wiring, "_fit_group", lambda *args: pytest.fail("recomputed a completed fold"))
    repeated = stage2_wiring.fit_models(**arguments)
    for a, b in zip(resumed.scoring, repeated.scoring):
        np.testing.assert_array_equal(a.posterior_draws.tile(0, 3), b.posterior_draws.tile(0, 3))


def test_failed_checkpoint_write_does_not_destroy_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "model0.npz"
    fit = _fit(1)
    save_checkpoint(path, fit, "key", 1.0)
    def interrupted(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(np, "savez", interrupted)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(path, _fit(2), "key", 2.0)
    _equal(fit, load_checkpoint(path, "key"))


def test_a_group_whose_fit_raises_is_quarantined_and_the_other_groups_are_fitted(tmp_path, monkeypatch):
    # Three quantitative models on different training rows: three groups. The middle group's fit has no certified
    # fixed point; its model comes back as its covariates alone with the failure among its refusals and its outer
    # criterion unmet (model_acceptance quarantines it), and the request still returns every model.
    monkeypatch.setattr(stage2_wiring, "store_measurement", lambda *args: None)
    failing = stage2_wiring._seed(91, 1)

    def engine(*args):
        if args[7] == failing:
            raise FloatingPointError("a starting point has no certified fixed point")
        fit = _fit(args[7])
        # A fit's state digest is its 32 bytes per model, as the null model's (the stub certificate's is per model).
        certificate = dataclasses.replace(fit.certificate, state_digest=np.zeros((1, 32), dtype=np.uint8))
        return [dataclasses.replace(fit, scoring=dataclasses.replace(fit.scoring, trait_type=TraitType.QUANTITATIVE), certificate=certificate)]

    monkeypatch.setattr(stage2_wiring, "_fit_group", engine)
    rng = np.random.default_rng(5)
    training = np.ones((12, 3), dtype=bool)
    training[np.arange(3), np.arange(3)] = False
    fitted = stage2_wiring.fit_models(
        store=SimpleNamespace(), store_columns=np.arange(12), covariates=np.column_stack([np.ones(12), rng.normal(size=12)]),
        covariate_columns=np.ones((3, 2), dtype=bool), targets=rng.normal(size=(12, 3)), training=training,
        trait_types=(TraitType.QUANTITATIVE,) * 3, log_variance_offset=None, budget=_budget(), work_dir=tmp_path, seed=91,
        draw_count=8, checkpoint_key="problem",
    )
    assert len(fitted.scoring) == 3 and np.asarray(fitted.certificate.outer_criterion_met).tolist() == [True, False, True]
    assert fitted.scoring[1].store_rows.shape[0] == 0
    assert any(reason.startswith("model 1: models 1: the fit failed: a starting point has no certified fixed point") for reason in fitted.certificate.refusals)
    # The failure is checkpointed like a finished fit, so a rerun does not fit the group again.
    monkeypatch.setattr(stage2_wiring, "_fit_group", lambda *args: pytest.fail("refitted a checkpointed group"))
    repeated = stage2_wiring.fit_models(
        store=SimpleNamespace(), store_columns=np.arange(12), covariates=np.column_stack([np.ones(12), rng.normal(size=12)]),
        covariate_columns=np.ones((3, 2), dtype=bool), targets=rng.normal(size=(12, 3)), training=training,
        trait_types=(TraitType.QUANTITATIVE,) * 3, log_variance_offset=None, budget=_budget(), work_dir=tmp_path, seed=91,
        draw_count=8, checkpoint_key="problem",
    )
    assert np.asarray(repeated.certificate.outer_criterion_met).tolist() == [True, False, True]
