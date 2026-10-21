"""Warm starts from saved models (``warm_seeds``): what a seed maps onto a new fit's prior and members, and on the
synthetic store (``test_full_data_fit._store``) that a refit from a fit's own saved model starts at its fixed point
and that the other arm's fit started from it reaches the fit its cold start reaches. Machinery checks of the start,
not an accuracy claim."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from sv_pgs import (
    content_identity,
    fit_model,
    full_data_fit,
    stage2_wiring,
    warm_seeds,
    workspace_pipeline,
)
from sv_pgs.artifact import load_model, save_model
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType, VariantClass
from sv_pgs.dosage_store import (
    VARIANT_CLASSES,
    DosageStore,
    open_column,
    variant_column_directory,
)
from sv_pgs.draw_laws import ProductMixtureDraws
from sv_pgs.fast_scoring import ScoringPlan, score_genetic
from sv_pgs.scale_mixture_ep import (
    AnnotationGroup,
    Cavity,
    MixtureHyperparameters,
    class_log_density,
    halved_lattice,
    log_scale,
    scale_mixture_prior,
    tilted_moments,
)
from sv_pgs.warm_seeds import Seed, content_digest, for_problem, read_seeds, seed_lattice, warm_start
from tests.test_full_data_fit import _store, _StoreCodes

_TOLERANCE = 0.5 / fit_model.DRAW_COUNT
_hook = pytest.mark.skipif(
    "seeds" not in fit_model.FitRequest.__dataclass_fields__, reason="the fit's warm-start hook (FitRequest.seeds, the EB side's patch) is not in this tree",
)
_WORKING_BYTES = 1 << 26
_CHROMOSOMES = (("chr21", 180), ("chr22", 140))
"""``test_full_data_fit._store``'s chromosomes and their record counts, in store order."""


def _budget() -> ComputeBudget:
    return ComputeBudget(
        device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), host_bytes=1 << 31, cpu_threads=2
    )


def _prior(classes: np.ndarray, nodes: np.ndarray) -> object:
    """A prior over ``classes`` with one annotation column, on ``nodes``."""
    design = np.random.default_rng(3).standard_normal((classes.shape[0], 1))
    return scale_mixture_prior(
        class_index=classes, log_variance_offset=np.zeros(classes.shape[0]), annotation_design=design,
        annotation_groups=(AnnotationGroup(columns=np.array([0]), penalty=np.eye(1)),), nodes=nodes, floor=float(nodes[0]) - 1.0, top=float(nodes[-1]),
    )


def _hyperparameters(prior) -> MixtureHyperparameters:
    generator = np.random.default_rng(5)
    return MixtureHyperparameters(
        coefficients=0.3 * generator.standard_normal(prior.coefficient_size), log_smoothing=np.linspace(-1.0, 1.0, len(prior.smoothing_blocks)),
    )


def _seed(prior, hyperparameters, rows, scales, means, **changes) -> Seed:
    """The seed a fit of ``prior`` at ``hyperparameters`` would save."""
    fields = {
        "source": "saved", "model_name": "trait/fold0", "store_rows": rows, "signed_scales": scales, "means": means,
        "class_index": np.asarray(prior.class_index, dtype=np.int64), "log_scale": log_scale(prior, hyperparameters.coefficients),
        "log_density": class_log_density(prior, hyperparameters.coefficients), "log_variance_grid": np.asarray(prior.log_variance_grid, dtype=np.float64),
        "hyperparameters": hyperparameters, "noise": 0.7, "sites_digest": "sites", "problem_digest": "problem", "offset_digest": "offsets",
    }
    fields.update(changes)
    return Seed(**fields)


_NODES = np.linspace(np.log(1e-5), np.log(0.5), 9)
_ROWS = np.arange(0, 24, 2, dtype=np.int64)
"""Twelve members, every other store row of 30; class 1 (a deletion) every third member."""
_CLASSES = (np.arange(12) % 3 == 1).astype(np.int64)


def _variant_class() -> np.ndarray:
    values = np.full(30, VARIANT_CLASSES.index(VariantClass.SNV), dtype=np.uint8)
    values[_ROWS[_CLASSES == 1]] = VARIANT_CLASSES.index(VariantClass.DELETION)
    return values


def test_a_seed_of_this_prior_starts_at_its_own_hyperparameters_and_raw_effects() -> None:
    prior = _prior(_CLASSES, _NODES)
    hyperparameters = _hyperparameters(prior)
    generator = np.random.default_rng(7)
    scales, means, new_scales = generator.uniform(0.2, 0.6, 12), generator.standard_normal(12), generator.uniform(0.2, 0.6, 12)
    start = warm_start(_seed(prior, hyperparameters, _ROWS, scales, means), prior, _ROWS, new_scales, _variant_class())
    assert start is not None and start.noise == 0.7
    np.testing.assert_array_equal(start.hyperparameters.coefficients, hyperparameters.coefficients)
    np.testing.assert_array_equal(start.hyperparameters.log_smoothing, hyperparameters.log_smoothing)
    # Each member keeps its raw effect: its standardized mean times the new code sd over the seed's.
    np.testing.assert_allclose(start.mean, means * new_scales / scales, rtol=1e-15)
    assert "12 of this fit's 12 members mapped; its own hyperparameters;" in start.note


def test_a_seed_on_another_lattice_is_carried_to_this_one_and_unfitted_members_start_at_zero() -> None:
    prior = _prior(_CLASSES, _NODES)
    hyperparameters = _hyperparameters(prior)
    fine, fine_hyperparameters = halved_lattice(prior, hyperparameters)
    # The new fit also has six members the seed did not fit (the SNVs at rows 1, 3, ..., 11).
    rows = np.sort(np.concatenate([_ROWS, np.arange(1, 12, 2)]))
    _labels, classes = np.unique(_variant_class()[rows], return_inverse=True)
    wider = _prior(classes.astype(np.int64), np.asarray(fine.log_variance_grid))
    components = np.stack([np.arange(1.0, 13.0), -np.arange(1.0, 13.0)])
    seed = _seed(prior, hyperparameters, _ROWS, np.full(12, 0.5), np.arange(1.0, 13.0), components=components, component_noises=np.array([0.7, 0.9]), settled=True)
    start = warm_start(seed, wider, rows, np.ones(rows.shape[0]), _variant_class())
    # Six members the seed never fitted: not its own problem, so its settled mode search does not stand here.
    assert start is not None and not start.settled
    shared = np.isin(rows, _ROWS)
    np.testing.assert_array_equal(start.mean[shared], 2.0 * np.arange(1.0, 13.0))
    assert np.all(start.mean[~shared] == 0.0)
    # The mode mixture's components map as the means do: each raw effect kept, zero where the seed fitted no member.
    np.testing.assert_array_equal(start.components[:, shared], 2.0 * components)
    assert start.components.shape == (2, rows.shape[0]) and np.all(start.components[:, ~shared] == 0.0)
    # Each component keeps its own fixed point's noise variance.
    np.testing.assert_array_equal(start.component_noises, [0.7, 0.9])
    assert start.component_noises is not seed.component_noises
    # Another design over more members is another schema: the density alone, the seed's largest-shared class.
    assert "12 of this fit's 18 members mapped; the density of its class 0 alone;" in start.note
    # Its pooled density is that class's, continued onto the refined lattice; every other block at its edge.
    density = class_log_density(wider, start.hyperparameters.coefficients)
    np.testing.assert_allclose(density[0], density[1], atol=1e-12)
    np.testing.assert_allclose(density[0], class_log_density(fine, fine_hyperparameters.coefficients)[0], atol=1e-9)
    assert start.hyperparameters.log_smoothing[0] == hyperparameters.log_smoothing[0]
    assert np.all(np.isinf(start.hyperparameters.log_smoothing[1:]))
    # The same schema on the refined lattice takes the seed's own hyperparameters, carried as the lattice check does.
    carried = warm_start(seed, fine, _ROWS, np.ones(12), _variant_class())
    assert carried is not None and "its own hyperparameters carried from its 9-node lattice" in carried.note
    # Its own members on its own prior: its settled search stands. Its own prior with one member another column (an
    # SNV the seed never fitted) is another problem, the near duplicates its modes come from changed, so the search
    # runs again.
    assert carried.settled
    swapped = np.append(_ROWS[:-1], 25)
    other = warm_start(seed, prior, swapped, np.ones(12), _variant_class())
    assert other is not None and "11 of this fit's 12 members mapped; its own hyperparameters;" in other.note and not other.settled
    np.testing.assert_array_equal(carried.hyperparameters.coefficients, fine_hyperparameters.coefficients)


def test_a_seed_whose_densities_are_not_this_prior_s_starts_from_its_density_alone() -> None:
    prior = _prior(_CLASSES, _NODES)
    hyperparameters = _hyperparameters(prior)
    shifted = log_scale(prior, hyperparameters.coefficients) + 0.1
    start = warm_start(_seed(prior, hyperparameters, _ROWS, np.ones(12), np.ones(12), log_scale=shifted), prior, _ROWS, np.ones(12), _variant_class())
    assert start is not None and "the density of its class 0 alone" in start.note
    np.testing.assert_allclose(
        class_log_density(prior, start.hyperparameters.coefficients)[1], class_log_density(prior, hyperparameters.coefficients)[0], atol=1e-10,
    )


def test_the_seed_s_means_are_the_main_fixed_point_s_of_the_saved_mixture() -> None:
    """``main_means`` against the prior's own tilted moments at the first component's natural parameters, a tile of
    rows at a time; the other component (another mode) does not enter."""
    prior = _prior(_CLASSES, _NODES)
    hyperparameters = _hyperparameters(prior)
    generator = np.random.default_rng(11)
    shift, omega = generator.standard_normal((2, 12)) * 3.0, generator.uniform(10.0, 400.0, (2, 12))
    law = ProductMixtureDraws(
        class_index=np.asarray(prior.class_index, dtype=np.int64), log_scale=log_scale(prior, hyperparameters.coefficients),
        log_density=class_log_density(prior, hyperparameters.coefficients), log_variance_grid=np.asarray(prior.log_variance_grid, dtype=np.float64),
        shift=shift, omega=omega, weights=np.array([0.25, 0.75]), key=(1, 2), draw_count=4,
    )
    expected = tilted_moments(prior, hyperparameters, Cavity(precision=omega[0], shift=shift[0]), _WORKING_BYTES, np).mean
    np.testing.assert_allclose(warm_seeds.main_means(law, _WORKING_BYTES), expected, rtol=1e-12, atol=1e-15)
    # Five rows a tile: the same means.
    np.testing.assert_allclose(warm_seeds.main_means(law, 5 * (law.tile_row_bytes() + 8 * law.shape[1])), expected, rtol=1e-12, atol=1e-15)
    # A component's noise variance from the first's: omega_c is the members' squares over sigma_c^2, so a component
    # whose omega is half the first's at every member has twice its noise; the first's is its own.
    halved = dataclasses.replace(law, omega=np.stack([omega[0], 0.5 * omega[0]]))
    assert warm_seeds.component_noise(halved, 1, 0.7) == pytest.approx(1.4, rel=1e-12)
    assert warm_seeds.component_noise(halved, 0, 0.7) == 0.7


def test_a_seed_of_another_store_is_no_start() -> None:
    prior = _prior(_CLASSES, _NODES)
    seed = _seed(prior, _hyperparameters(prior), _ROWS + 20, np.ones(12), np.ones(12))
    assert warm_start(seed, prior, _ROWS, np.ones(12), _variant_class()) is None


def test_a_seed_is_checked_when_it_is_made() -> None:
    prior = _prior(_CLASSES, _NODES)
    hyperparameters = _hyperparameters(prior)
    with pytest.raises(ValueError, match="sorted"):
        _seed(prior, hyperparameters, _ROWS[::-1].copy(), np.ones(12), np.ones(12))
    with pytest.raises(ValueError, match="scales must be positive"):
        _seed(prior, hyperparameters, _ROWS, -np.ones(12), np.ones(12))
    with pytest.raises(ValueError, match="means"):
        _seed(prior, hyperparameters, _ROWS, np.ones(12), np.full(12, np.nan))
    with pytest.raises(ValueError, match="noise"):
        _seed(prior, hyperparameters, _ROWS, np.ones(12), np.ones(12), noise=0.0)
    with pytest.raises(ValueError, match="mixture components"):
        _seed(prior, hyperparameters, _ROWS, np.ones(12), np.ones(12), components=np.ones((2, 11)))
    with pytest.raises(ValueError, match="positive noise variance"):
        _seed(prior, hyperparameters, _ROWS, np.ones(12), np.ones(12), components=np.ones((2, 12)))
    with pytest.raises(ValueError, match="positive noise variance"):
        _seed(prior, hyperparameters, _ROWS, np.ones(12), np.ones(12), components=np.ones((2, 12)), component_noises=np.array([0.7, 0.0]))


def test_a_settled_search_stands_only_for_the_seed_s_own_problem() -> None:
    """A seed's settled mode search is for its own posterior (Codex EB review 11): for another problem, offsets or store
    sites it is cleared, its components still the mixture's start, so the fit searches for its modes again."""
    prior = _prior(_CLASSES, _NODES)
    components = np.ones((2, 12))
    seed = _seed(
        prior, _hyperparameters(prior), _ROWS, np.ones(12), np.ones(12), components=components, component_noises=np.array([0.7, 0.8]), settled=True,
    )
    assert for_problem(seed, "problem", "offsets", "sites") is seed
    assert warm_start(seed, prior, _ROWS, np.ones(12), _variant_class()).settled
    for other in (("another", "offsets", "sites"), ("problem", "another", "sites"), ("problem", "offsets", "another")):
        moved = for_problem(seed, *other)
        assert not moved.settled
        np.testing.assert_array_equal(moved.components, components)
        start = warm_start(moved, prior, _ROWS, np.ones(12), _variant_class())
        assert start is not None and not start.settled and start.components is not None


def test_a_seed_read_from_memory_is_named_by_its_content() -> None:
    """A seed with no saved model's digest is named by everything a fit takes from it (Codex EB review 12): two read
    from memory under one source but with other means are two seeds, and the same content is one."""
    prior = _prior(_CLASSES, _NODES)
    hyperparameters = _hyperparameters(prior)
    first = _seed(prior, hyperparameters, _ROWS, np.ones(12), np.ones(12), source="previous")
    again = _seed(prior, hyperparameters, _ROWS, np.ones(12), np.ones(12), source="previous")
    other = _seed(prior, hyperparameters, _ROWS, np.ones(12), np.full(12, 2.0), source="previous")
    assert content_digest(first) == content_digest(again) != content_digest(other)
    assert content_digest(first) != content_digest(dataclasses.replace(first, settled=True))


def test_a_group_fits_on_its_seeds_lattice_only_where_every_seed_shares_one_that_covers_the_start() -> None:
    prior = _prior(_CLASSES, _NODES)
    hyperparameters = _hyperparameters(prior)
    fine, _fine_hyperparameters = halved_lattice(prior, hyperparameters)
    seed = _seed(prior, hyperparameters, _ROWS, np.ones(12), np.ones(12))
    fine_seed = _seed(fine, _fine_hyperparameters, _ROWS, np.ones(12), np.ones(12))
    # The refined lattice covers the start's nodes at half their spacing.
    np.testing.assert_array_equal(seed_lattice([fine_seed], _NODES), fine.log_variance_grid)
    np.testing.assert_array_equal(seed_lattice([seed, seed], _NODES), _NODES)
    # A coarser lattice, one that misses the top, mixed lattices, or a model without a seed: the start's own.
    np.testing.assert_array_equal(seed_lattice([seed], fine.log_variance_grid), fine.log_variance_grid)
    np.testing.assert_array_equal(seed_lattice([seed], _NODES + 1.0), _NODES + 1.0)
    np.testing.assert_array_equal(seed_lattice([seed, fine_seed], _NODES), _NODES)
    np.testing.assert_array_equal(seed_lattice([fine_seed, None], _NODES), _NODES)


def _request_arguments(store, covariate, targets, offsets, work_dir: Path) -> dict:
    samples = store.n_samples
    training = np.arange(samples) < samples * 4 // 5
    return {
        "store": store, "store_columns": np.arange(samples, dtype=np.int64), "covariates": covariate[:, None], "covariate_names": ("covariate",),
        "covariate_columns": np.ones((1, 1), dtype=bool), "targets": np.where(training, targets, np.nan)[:, None], "training": training[:, None],
        "model_names": ("trait/fold0",), "trait_types": (TraitType.QUANTITATIVE,), "research_ids": tuple(f"person{index}" for index in range(samples)),
        "log_variance_offset": offsets, "budget": _budget(), "work_dir": work_dir, "seed": 3,
    }


class _Counted(full_data_fit._FullDataMeanField):
    """The sample route's oracle, each one made kept, so a fit's sweeps over the data are counted across its solves (the
    nested search's base, the annotated prior, each lattice refit)."""

    made: ClassVar[list] = []

    def __init__(self, *arguments, **keywords) -> None:
        super().__init__(*arguments, **keywords)
        _Counted.made.append(self)


def _fit(request: fit_model.FitRequest, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """``request``'s fit on the sample route (the reference route, pinned: timing alone chooses it) and the passes over
    the data every solve of it made."""
    monkeypatch.setattr(stage2_wiring, "pass_costs", lambda band, source, xp: (0.0, np.inf))
    monkeypatch.setattr(full_data_fit, "_FullDataMeanField", _Counted)
    _Counted.made = []
    model = fit_model.fit(request)
    return model, sum(oracle.passes for oracle in _Counted.made)


@pytest.fixture(scope="module")
def arms(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """The synthetic store with every eighth record a deletion (its structural records), and each arm's cold fit of one
    trait, saved: snv_indel leaves the deletions out (log reliability -inf), with_sv fits every record."""
    root = tmp_path_factory.mktemp("arms")
    _unused, covariate, targets, genetic = _store(root / "store", 7)
    structural = np.arange(sum(count for _name, count in _CHROMOSOMES)) % 8 == 3
    start = 0
    for chromosome, count in _CHROMOSOMES:
        values, _attributes = open_column(variant_column_directory(root / "store", chromosome, "variant_class"), writable=True)
        values[structural[start : start + count]] = VARIANT_CLASSES.index(VariantClass.DELETION)
        values.flush()
        del values
        start += count
    store = DosageStore.open(root / "store")
    assert np.array_equal(np.asarray(store.variant_table.variant_class) != 0, structural)
    offsets = {"snv_indel": np.where(structural, -np.inf, 0.0), "with_sv": np.zeros(structural.shape[0])}
    fits, passes = {}, {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        for arm, offset in offsets.items():
            fits[arm], passes[arm] = _fit(fit_model.FitRequest(**_request_arguments(store, covariate, targets, offset, root / "work" / arm)), monkeypatch)
            save_model(root / "models" / arm / "trait", fits[arm])
    return SimpleNamespace(root=root, store=store, covariate=covariate, targets=targets, genetic=genetic, offsets=offsets, fits=fits, passes=passes)


def _held_out_scores(arms: SimpleNamespace, model) -> np.ndarray:
    held_out = np.arange(arms.store.n_samples) >= arms.store.n_samples * 4 // 5
    return score_genetic(_StoreCodes(arms.store), ScoringPlan.from_models(list(model.scoring)), _budget()).means[held_out, 0]


def test_a_saved_model_is_a_seed_of_its_mean_field_model(arms: SimpleNamespace) -> None:
    saved = load_model(arms.root / "models" / "with_sv" / "trait")
    (seed,) = read_seeds(arms.root / "models" / "with_sv" / "trait", _WORKING_BYTES).values()
    scoring, law = saved.scoring[0], saved.scoring[0].posterior_draws
    np.testing.assert_array_equal(seed.store_rows, scoring.store_rows)
    # The main fixed point's means, which are the scorer's coefficients only where the mixture holds no other mode.
    if law.weights.shape[0] == 1:
        np.testing.assert_allclose(seed.means, scoring.coefficients, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(seed.log_variance_grid, law.log_variance_grid)
    np.testing.assert_array_equal(seed.hyperparameters.coefficients, saved.hyperparameters[0].coefficients)
    assert seed.noise == float(saved.noise_variance[0]) and seed.model_name == "trait/fold0"
    assert (seed.sites_digest, seed.problem_digest, seed.offset_digest) == (
        saved.provenance.sites_digest, saved.provenance.problem_digest, saved.provenance.offset_digest,
    )
    # The saved file's digest, for the engine's checkpoint key, and the mode mixture's components, the main one first.
    assert seed.digest == hashlib.sha256((arms.root / "models" / "with_sv" / "trait" / "model.json").read_bytes()).hexdigest()
    count = int(saved.certificate["mixture_components"][0])
    assert seed.components is not None and seed.components.shape == (count, seed.store_rows.shape[0])
    np.testing.assert_array_equal(seed.components[0], seed.means)
    assert seed.component_noises is not None and seed.component_noises.shape == (count,) and np.all(seed.component_noises > 0.0)
    assert seed.component_noises[0] == pytest.approx(seed.noise, rel=1e-12)
    assert seed.settled == (int(saved.certificate["budget_unresolved"][0]) == 0)
    # A directory that is no whole saved model is no seed (its reason logged), never an error.
    broken = arms.root / "broken"
    shutil.copytree(arms.root / "models" / "with_sv" / "trait", broken)
    (broken / "arrays.npz").unlink()
    assert read_seeds(broken, _WORKING_BYTES) == {}


@_hook
def test_a_fit_hands_on_a_settled_search_only_for_the_seed_s_own_problem(arms: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """``fit_model.fit`` gives the wiring each seed with its settled mode search where the request is the seed's own
    problem, offsets and store sites, and with it cleared where any differs, here the targets (Codex EB review 11). Its
    checkpoints name an in-memory seed by its content, so a seed with other means is another fit (review 12)."""
    (seed,) = read_seeds(arms.root / "models" / "with_sv" / "trait", _WORKING_BYTES).values()
    assert seed.settled
    handed = []

    class _Handed(Exception):
        pass

    def capture(**keywords):
        handed.append(keywords["seeds"])
        raise _Handed

    monkeypatch.setattr(fit_model, "fit_models", capture)
    work = arms.root / "work" / "settled"
    for targets in (arms.targets, arms.targets + 0.01):
        with pytest.raises(_Handed):
            fit_model.fit(fit_model.FitRequest(**_request_arguments(arms.store, arms.covariate, targets, arms.offsets["with_sv"], work), seeds=(seed,)))
    assert handed[0][0].settled and not handed[1][0].settled
    in_memory = dataclasses.replace(seed, digest="", source="previous")

    def named(chosen):
        request = fit_model.FitRequest(**_request_arguments(arms.store, arms.covariate, arms.targets, arms.offsets["with_sv"], work), seeds=(chosen,))
        return fit_model.fit_identity(request, "store")["seeds"]

    assert named(seed) == seed.digest
    assert named(in_memory) == named(dataclasses.replace(in_memory)) != named(dataclasses.replace(in_memory, means=1.01 * in_memory.means))


@_hook
def test_a_refit_from_its_own_saved_model_starts_at_its_fixed_point(arms: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """The with_sv arm refitted from its own saved model: the start is the saved state on the saved lattice, so the
    outer loop finds it stationary and returns the same fit for a fraction of the cold fit's passes."""
    seeds = tuple(read_seeds(arms.root / "models" / "with_sv" / "trait", _WORKING_BYTES).values())
    request = fit_model.FitRequest(
        **_request_arguments(arms.store, arms.covariate, arms.targets, arms.offsets["with_sv"], arms.root / "work" / "with_sv_warm"), seeds=seeds,
    )
    warm, passes = _fit(request, monkeypatch)
    cold = arms.fits["with_sv"]
    print(f"warm seeds: with_sv cold {arms.passes['with_sv']} passes, refit from its own saved model {passes} passes")
    assert any("its own hyperparameters" in refusal for refusal in warm.refusals)
    assert warm.certificate["remaining_gain"][0] <= _TOLERANCE
    assert bool(warm.certificate["outer_criterion_met"][0]) == bool(cold.certificate["outer_criterion_met"][0])
    np.testing.assert_array_equal(warm.scoring[0].store_rows, cold.scoring[0].store_rows)
    assert np.corrcoef(warm.scoring[0].coefficients, cold.scoring[0].coefficients)[0, 1] > 0.999
    assert np.corrcoef(_held_out_scores(arms, warm), _held_out_scores(arms, cold))[0, 1] > 0.999
    np.testing.assert_allclose(warm.noise_variance, cold.noise_variance, rtol=0.01)
    # The outer loop's first check finds the saved state stationary: what is left is the first solve from it and the mode
    # mixture's search, which every fit runs (93 passes cold and 36 seeded when this was written).
    assert passes < arms.passes["with_sv"] / 2


@_hook
def test_the_other_arm_started_from_this_arm_reaches_its_cold_fit(arms: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """with_sv started from snv_indel's saved model (the shared records' effects, the deletions' at zero, snv_indel's
    one class density as the pooled density) reaches with_sv's cold fit, as two fits of one group agree
    (``test_stage2_wiring``: held-out scores to 0.99, noise to 5%)."""
    seeds = tuple(read_seeds(arms.root / "models" / "snv_indel" / "trait", _WORKING_BYTES).values())
    request = fit_model.FitRequest(
        **_request_arguments(arms.store, arms.covariate, arms.targets, arms.offsets["with_sv"], arms.root / "work" / "with_sv_from_snv"), seeds=seeds,
    )
    warm, passes = _fit(request, monkeypatch)
    cold = arms.fits["with_sv"]
    print(f"warm seeds: with_sv cold {arms.passes['with_sv']} passes, from snv_indel's saved model {passes} passes")
    mapped = arms.fits["snv_indel"].scoring[0].store_rows.shape[0]
    assert any(f"{mapped:,} of this fit's {cold.scoring[0].store_rows.shape[0]:,} members mapped; the density of its class 0 alone" in refusal for refusal in warm.refusals)
    assert warm.certificate["remaining_gain"][0] <= _TOLERANCE
    np.testing.assert_array_equal(warm.scoring[0].store_rows, cold.scoring[0].store_rows)
    assert np.corrcoef(_held_out_scores(arms, warm), _held_out_scores(arms, cold))[0, 1] > 0.99
    np.testing.assert_allclose(warm.noise_variance, cold.noise_variance, rtol=0.05)
    assert passes < arms.passes["with_sv"]


def test_the_fit_step_starts_each_arm_from_the_best_saved_model_its_patterns_find(arms: SimpleNamespace) -> None:
    """Candidates rank by the same records' reliabilities, the same problem, acceptance, then age; an unreadable one
    is passed over for the next, and one of another store's sites is never a candidate."""
    mirror = arms.root / "mirror"
    saved = {arm: arms.root / "models" / arm / "trait" for arm in ("snv_indel", "with_sv")}
    places = {
        "own": (saved["with_sv"], mirror / "a" / "run" / "fit" / "models" / "with_sv" / "trait", 1_000),
        "other_arm": (saved["snv_indel"], mirror / "b" / "run" / "fit.partial" / "models" / "with_sv" / "trait", 2_000),
        "broken": (saved["with_sv"], mirror / "c" / "run" / "fit" / "models" / "with_sv" / "trait", 3_000),
        "other_sites": (saved["with_sv"], mirror / "d" / "run" / "fit" / "models" / "with_sv" / "trait", 4_000),
    }
    for source, target, modified in places.values():
        shutil.copytree(source, target)
        os.utime(target / "model.json", (modified, modified))
    (places["broken"][1] / "arrays.npz").unlink()
    metadata = json.loads((places["other_sites"][1] / "model.json").read_text())
    metadata["provenance"]["sites_digest"] = "another store"
    (places["other_sites"][1] / "model.json").write_text(json.dumps(metadata))
    pattern = str(mirror / "*" / "run" / "fit*" / "models" / "{arm}" / "{trait}")
    run = SimpleNamespace(config=SimpleNamespace(warm_start_models=(pattern,)), budget=_budget())
    cohort = SimpleNamespace(trait_names=("trait",), trait_models=lambda trait: np.array([0]))
    request = fit_model.FitRequest(**_request_arguments(arms.store, arms.covariate, arms.targets, arms.offsets["with_sv"], arms.root / "work" / "unused"))
    (seed,) = workspace_pipeline._warm_seeds(run, cohort, arms.store, request, "with_sv")
    assert seed is not None and seed.source == str(places["own"][1])
    # Without the own arm's model, the other arm's is the start; with no pattern, the fit is not seeded at all.
    shutil.rmtree(places["own"][1])
    (seed,) = workspace_pipeline._warm_seeds(run, cohort, arms.store, request, "with_sv")
    assert seed is not None and seed.source == str(places["other_arm"][1])
    assert workspace_pipeline._warm_seeds(run, cohort, arms.store, request, "snv_indel") == (None,)
    assert workspace_pipeline._warm_seeds(SimpleNamespace(config=SimpleNamespace(warm_start_models=()), budget=_budget()), cohort, arms.store, request, "with_sv") is None
    assert warm_seeds.seeds_from_model(load_model(saved["with_sv"]), "x", _WORKING_BYTES)["trait/fold0"].offset_digest == fit_model.offset_digest(arms.offsets["with_sv"])
    # The problem the fit step ranks by is the one fit_model.fit records.
    assert workspace_pipeline._request_problem(request) == load_model(saved["with_sv"]).provenance.problem_digest


def test_a_saved_model_read_through_the_mounted_bucket_is_a_seed(arms: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """A round's warm start reads the earlier run's model through the job's gcsfuse mount of the ops bucket
    (/mnt/disks/ops, ``content_identity.GCS_MOUNTS``), where ``content_identity.file_digest`` is the object's
    generation. The model's checksum is of its bytes, written on the saving job's own disk and checked on the mount
    alike, so the model is a seed and no object's metadata is asked for; a damaged copy is passed over for the next."""
    mount = (arms.root / "ops").resolve()
    saved = arms.root / "models" / "with_sv" / "trait"
    good = mount / "svpgs" / "base" / "run" / "fit" / "models" / "with_sv" / "trait"
    damaged = mount / "svpgs" / "other" / "run" / "fit" / "models" / "with_sv" / "trait"
    for target, modified in ((good, 1_000), (damaged, 2_000)):
        shutil.copytree(saved, target)
        os.utime(target / "model.json", (modified, modified))
    content = bytearray((damaged / "arrays.npz").read_bytes())
    content[len(content) // 2] ^= 0xFF
    (damaged / "arrays.npz").write_bytes(bytes(content))
    asked: list[str] = []
    blob = SimpleNamespace(generation=1, size=len(content), crc32c="AAAAAA==")
    client = SimpleNamespace(bucket=lambda bucket, user_project=None: SimpleNamespace(get_blob=lambda name: asked.append(name) or blob))
    monkeypatch.setattr(content_identity, "_storage_client", lambda: client)
    monkeypatch.setenv(content_identity.GCS_MOUNTS, json.dumps({str(mount): "gs://ops-bucket"}))
    metadata = json.loads((good / "model.json").read_text())
    assert metadata["arrays_sha256"] == hashlib.sha256((good / "arrays.npz").read_bytes()).hexdigest()
    assert read_seeds(good, _WORKING_BYTES) and read_seeds(damaged, _WORKING_BYTES) == {}
    run = SimpleNamespace(config=SimpleNamespace(warm_start_models=(str(mount / "svpgs" / "*" / "run" / "fit*" / "models" / "{arm}" / "{trait}"),)), budget=_budget())
    cohort = SimpleNamespace(trait_names=("trait",), trait_models=lambda trait: np.array([0]))
    request = fit_model.FitRequest(**_request_arguments(arms.store, arms.covariate, arms.targets, arms.offsets["with_sv"], arms.root / "work" / "unused"))
    (seed,) = workspace_pipeline._warm_seeds(run, cohort, arms.store, request, "with_sv")
    assert seed is not None and seed.source == str(good) and asked == []
