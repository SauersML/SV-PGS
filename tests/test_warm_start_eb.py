"""The empirical Bayes side of a warm start (``stage2_wiring._fit_group``'s seeds, ``full_data_fit.MixtureStart``): a fit
from its own saved model is the same fit in a handful of sweeps, its mode search standing on the saved components, and
a seeded fit with no certified start is fitted from the prior. The arms and their saved cold fits are
``test_warm_seeds``'s."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from sv_pgs import fit_model, full_data_fit, stage2_wiring
from sv_pgs.warm_seeds import read_seeds
from tests.test_warm_seeds import _WORKING_BYTES, _held_out_scores, _request_arguments, arms  # noqa: F401 (``arms`` is the module's fixture)


class _Sweeps(full_data_fit._FullDataMeanField):
    """The sample route's oracle, each one kept: the empirical Bayes' (those with an allowance) and the mode mixture's,
    whose orders from zero carry an ``order_seed``."""

    made: ClassVar[list] = []
    orders: ClassVar[int] = 0

    def __init__(self, *arguments, **keywords) -> None:
        super().__init__(*arguments, **keywords)
        _Sweeps.made.append(self)
        _Sweeps.orders += "order_seed" in keywords


def _fit(request: fit_model.FitRequest, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """``request``'s fit on the sample route and its sweeps: the empirical Bayes' and the mode mixture's."""
    monkeypatch.setattr(stage2_wiring, "pass_costs", lambda band, source, xp: (0.0, np.inf))
    monkeypatch.setattr(full_data_fit, "_FullDataMeanField", _Sweeps)
    _Sweeps.made, _Sweeps.orders = [], 0
    model = fit_model.fit(request)
    eb = sum(int(oracle.model_passes.sum()) for oracle in _Sweeps.made if oracle.pass_allowance is not None)
    mixture = sum(int(oracle.model_passes.sum()) for oracle in _Sweeps.made if oracle.pass_allowance is None)
    return model, eb, mixture, _Sweeps.orders


def test_a_refit_from_its_own_saved_model_is_the_same_fit_in_a_few_sweeps(arms: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """The acceptance test of the warm start's empirical Bayes: from a converged fit's saved model (its hyperparameters,
    noise, main fixed point and mode mixture, each component at its own noise), the fit reaches the same fixed point
    within the draws' resolution in at most 10 sweeps of its empirical Bayes, and its mode search stands on the saved
    components (each solved once from where it was a fixed point: no order from zero), where the cold fit searched."""
    cold_request = fit_model.FitRequest(**_request_arguments(arms.store, arms.covariate, arms.targets, arms.offsets["with_sv"], arms.root / "work" / "with_sv_counted"))
    cold, cold_eb, cold_mixture, cold_orders = _fit(cold_request, monkeypatch)
    seeds = tuple(read_seeds(arms.root / "models" / "with_sv" / "trait", _WORKING_BYTES).values())
    (seed,) = seeds
    request = fit_model.FitRequest(
        **_request_arguments(arms.store, arms.covariate, arms.targets, arms.offsets["with_sv"], arms.root / "work" / "with_sv_eb_warm"), seeds=seeds,
    )
    warm, eb, mixture, orders = _fit(request, monkeypatch)
    components = seed.components.shape[0]
    print(
        f"warm start: with_sv cold {cold_eb} empirical Bayes and {cold_mixture} mode mixture sweeps ({cold_orders} orders); from its own saved "
        f"model ({components} components, settled {seed.settled}) {eb} and {mixture} sweeps ({orders} orders)"
    )
    assert seed.settled and cold_orders > 0
    assert eb <= 10 and orders == 0 and mixture <= 3 * (components - 1)
    assert bool(warm.certificate["outer_criterion_met"][0]) == bool(cold.certificate["outer_criterion_met"][0])
    np.testing.assert_allclose(warm.noise_variance, cold.noise_variance, rtol=1e-3)
    assert np.corrcoef(warm.scoring[0].coefficients, cold.scoring[0].coefficients)[0, 1] > 0.9999
    assert np.corrcoef(_held_out_scores(arms, warm), _held_out_scores(arms, cold))[0, 1] > 0.9999


def test_a_seeded_fit_with_no_certified_start_is_fitted_from_the_prior(arms: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """A seed moves where the fit starts, never whether it is fitted: where the seeded start has no certified fixed
    point the group is fitted from the prior, as without its seed, and the certificate says so."""
    seeds = tuple(read_seeds(arms.root / "models" / "with_sv" / "trait", _WORKING_BYTES).values())
    fit_full_data = stage2_wiring.fit_full_data
    calls: list[set[str]] = []

    def refused_once(**keywords):
        calls.append(set(keywords))
        if len(calls) == 1:
            raise FloatingPointError("no certified fixed point from this start (the test's)")
        return fit_full_data(**keywords)

    monkeypatch.setattr(stage2_wiring, "fit_full_data", refused_once)
    request = fit_model.FitRequest(
        **_request_arguments(arms.store, arms.covariate, arms.targets, arms.offsets["with_sv"], arms.root / "work" / "with_sv_refused"), seeds=seeds,
    )
    fallen, _eb, _mixture, _orders = _fit(request, monkeypatch)
    # The seeded start (its continuation and mode mixture), then the fit from the prior.
    assert {"starts", "start_noise", "start_mean", "mixture_start"} <= calls[0] and "starts" not in calls[1]
    assert any("so it was fitted from the prior: no certified fixed point from this start (the test's)" in refusal for refusal in fallen.refusals)
    cold = arms.fits["with_sv"]
    assert np.corrcoef(_held_out_scores(arms, fallen), _held_out_scores(arms, cold))[0, 1] > 0.99
    np.testing.assert_allclose(fallen.noise_variance, cold.noise_variance, rtol=0.05)


def test_a_seeded_fit_whose_fit_from_the_prior_makes_no_certified_progress_is_scored_uncertified(arms: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Where neither the seeded start's empirical Bayes nor the fit from the prior makes certified progress (each outer
    loop raises ``NoCertifiedProgress``), the model is still fitted, scored and flagged: the fit from the prior returns
    its best-ELBO state, uncertified, and its certificate names both endings (``full_data_fit.fit_full_data``)."""
    from sv_pgs.scale_mixture_ep import NoCertifiedProgress

    seeds = tuple(read_seeds(arms.root / "models" / "with_sv" / "trait", _WORKING_BYTES).values())
    fit_full_data = stage2_wiring.fit_full_data
    continued: list[bool] = []

    def recorded(**keywords):
        continued.append("starts" in keywords)
        return fit_full_data(**keywords)

    def walking(prior, starts, oracle, working_bytes, tolerance):
        # The start's solve, then the loop's error.
        assert all(point is not None for point in oracle(list(starts)))
        raise NoCertifiedProgress("the Newton-B step makes no certified progress at the EP fixed point (the test's)")

    monkeypatch.setattr(stage2_wiring, "fit_full_data", recorded)
    monkeypatch.setattr(full_data_fit, "fit_hyperparameters", walking)
    request = fit_model.FitRequest(
        **_request_arguments(arms.store, arms.covariate, arms.targets, arms.offsets["with_sv"], arms.root / "work" / "with_sv_walking"), seeds=seeds,
    )
    walked, _eb, _mixture, _orders = _fit(request, monkeypatch)
    # The seeded start, then the fit from the prior (a lattice check's refit, if any, raises too and leaves it standing).
    assert continued[:2] == [True, False]
    assert not bool(walked.certificate["outer_criterion_met"][0])
    assert any(
        "so it was fitted from the prior: the Newton-B step makes no certified progress at the EP fixed point (the test's)" in refusal
        for refusal in walked.refusals
    )
    assert any("the outer loop made no certified progress, and every model returns its best-ELBO state, uncertified" in refusal for refusal in walked.refusals)
    scores = _held_out_scores(arms, walked)
    assert np.all(np.isfinite(scores)) and float(np.std(scores)) > 0.0


def test_a_seed_keys_the_completed_models_but_not_the_starts_a_retry_resumes(arms: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """The seeds name the fit's checkpoints (a completed model is the fit from its start), not the restart store's
    empirical Bayes starts (``request_key``), which a preempted fit resumes whatever seed its retry chose; a request
    whose every seed is None keeps the cold fit's identity."""
    seeds = tuple(read_seeds(arms.root / "models" / "with_sv" / "trait", _WORKING_BYTES).values())
    keys: list[tuple[str, str]] = []

    class _Keyed(Exception):
        pass

    def keyed(**keywords):
        keys.append((keywords["checkpoint_key"], keywords["request_key"]))
        raise _Keyed

    monkeypatch.setattr(fit_model, "fit_models", keyed)
    for chosen in (None, seeds):
        with pytest.raises(_Keyed):
            fit_model.fit(fit_model.FitRequest(
                **_request_arguments(arms.store, arms.covariate, arms.targets, arms.offsets["with_sv"], arms.root / "work" / "with_sv_keys"), seeds=chosen,
            ))
    (cold_checkpoint, cold_request), (seeded_checkpoint, seeded_request) = keys
    assert seeded_checkpoint != cold_checkpoint and seeded_request == cold_request
    unseeded = fit_model.FitRequest(
        **_request_arguments(arms.store, arms.covariate, arms.targets, arms.offsets["with_sv"], arms.root / "work" / "with_sv_keys"), seeds=(None,),
    )
    assert "seeds" not in fit_model.fit_identity(unseeded, "store")
