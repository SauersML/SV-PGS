"""The EB trajectory monitor (``sv_pgs.eb_monitor``): it scores each mean-field call's posterior mean on the evaluation
rows from codes Stage 0's own pass kept, its metrics are the report's, and the fit it watches is the fit without it."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sv_pgs import eb_monitor
from sv_pgs.config import TraitType
from sv_pgs.eb_monitor import EbMonitor, EvaluationCodes, EvaluationRows, GroupEvaluation, partial_r2
from tests.test_full_data_fit import _store
from tests.test_stage2_wiring import _budget


def test_the_monitor_s_metrics_are_the_report_s() -> None:
    """The partial r^2 is the report's ``pgs_partial_r2`` and the AUC its ``pgs_auc``, on the same rows and covariates."""
    from sv_pgs.held_out_comparison import auc_components
    from sv_pgs.workspace_pipeline import _partial_r2

    generator = np.random.default_rng(4)
    covariates = np.column_stack([np.ones(300), generator.standard_normal((300, 3))])
    score = generator.standard_normal(300)
    target = 0.3 * score + covariates @ np.array([0.5, 1.0, -0.4, 0.2]) + generator.standard_normal(300)
    assert partial_r2(target, score, covariates) == pytest.approx(_partial_r2(target, score, covariates), rel=1e-12)
    labels = (target > np.median(target)).astype(np.float64)
    assert eb_monitor.auc(labels, score) == pytest.approx(auc_components(labels, score).auc, rel=1e-12)


def _monitor(capsys: pytest.CaptureFixture[str]) -> tuple[EbMonitor, np.ndarray, np.ndarray, np.ndarray, GroupEvaluation]:
    """A monitor over 40 members on 60 evaluation rows of a disease, its codes filled as Stage 0's tiles fill them:
    (monitor, signed codes [members, rows], Stage 0's means and scales, the evaluation)."""
    generator = np.random.default_rng(9)
    candidates = np.arange(0, 100, 2, dtype=np.int64)
    members = candidates[np.sort(generator.choice(candidates.shape[0], 40, replace=False))]
    columns = np.sort(generator.choice(200, 60, replace=False)).astype(np.int64)
    tiles = generator.integers(0, 255, size=(candidates.shape[0], 200), dtype=np.uint8)
    codes = EvaluationCodes(candidates, columns, 16)
    for start in range(0, candidates.shape[0], 16):
        codes.capture(candidates[start:start + 16], tiles[start:start + 16])
    signed = tiles[np.searchsorted(candidates, members)][:, columns].astype(np.float64) - 127.0
    means, scales = signed.mean(axis=1) + 0.1, signed.std(axis=1) + 0.5
    evaluation = GroupEvaluation(
        store_columns=columns, covariates=np.column_stack([np.ones(60), generator.standard_normal(60)]),
        targets=np.column_stack([(generator.random(60) < 0.4).astype(np.float64)]), disease=np.array([True]), names=("trait/fold0",),
    )
    monitor = EbMonitor.start(codes, members, means, scales, evaluation)
    assert monitor is not None
    capsys.readouterr()
    return monitor, signed, means, scales, evaluation


def test_a_score_is_the_standardized_codes_times_the_posterior_mean(capsys: pytest.CaptureFixture[str]) -> None:
    """The monitor's score is (c - mu) / s . m over the members, from the codes the tiles left, and each call is
    logged with its ELBO, the coefficients' move since the stage's last call, and the metrics of that score."""
    from sv_pgs.scale_mixture_ep import MixtureHyperparameters

    monitor, signed, means, scales, evaluation = _monitor(capsys)
    generator = np.random.default_rng(1)
    mean = generator.standard_normal((40, 1))
    expected = ((signed - means[:, None]) / scales[:, None]).T @ mean
    np.testing.assert_allclose(monitor.scores(mean), expected, rtol=1e-5, atol=1e-5)
    monitor.stage("base")
    coefficients = [np.array([0.5, -1.0]), np.array([0.8, -1.4])]
    for coefficients_now in coefficients:
        oracle = SimpleNamespace(mean=mean, elbo=np.array([-123.25]))
        monitor.view([0]).observe(oracle, [MixtureHyperparameters(coefficients=coefficients_now, log_smoothing=np.zeros(0))], [0])
    lines = [line for line in capsys.readouterr().err.splitlines() if "eb monitor: model" in line]
    assert len(lines) == 2
    score = expected[:, 0]
    r2 = partial_r2(evaluation.targets[:, 0], score, evaluation.covariates)
    area = eb_monitor.auc(evaluation.targets[:, 0], score)
    first = re.search(r"model trait/fold0 iter 1 elbo -123.25 pgs_partial_r2 (\S+) pgs_auc (\S+) x_step - stage base", lines[0])
    assert first is not None
    assert float(first.group(1)) == pytest.approx(r2, rel=1e-4) and float(first.group(2)) == pytest.approx(area, rel=1e-4)
    assert f"x_step {np.linalg.norm(coefficients[1] - coefficients[0]):.4g} stage base" in lines[1]


def test_the_monitor_keeps_to_its_share_of_the_fit_s_time(capsys: pytest.CaptureFixture[str]) -> None:
    """A call is scored only while the scores so far and the last one's time fit MONITOR_SHARE of the fit's time since
    the monitor began; one that does not is still logged, its ELBO and move without metrics."""
    from sv_pgs.scale_mixture_ep import MixtureHyperparameters

    monitor, _signed, _means, _scales, _evaluation = _monitor(capsys)
    monitor._spent = 1e9
    oracle = SimpleNamespace(mean=np.ones((40, 1)), elbo=np.array([-1.0]))
    monitor.view([0]).observe(oracle, [MixtureHyperparameters(coefficients=np.zeros(2), log_smoothing=np.zeros(0))], [0])
    line = next(line for line in capsys.readouterr().err.splitlines() if "eb monitor: model" in line)
    assert "not scored" in line and "pgs_partial_r2" not in line and "elbo -1 " in line


def test_the_codes_are_a_ledger_cache_that_yields_and_never_outlives_them(capsys: pytest.CaptureFixture[str]) -> None:
    """The codes hold a cache lease of the host ledger: evicted, they are off (one line); dropped without a release
    (a group's fit that ended early or failed), their lease goes with them."""
    import gc

    from sv_pgs.memory_broker import HOST, memory_scope

    def purposes(broker) -> list[str]:
        return [lease.purpose for lease in broker.leases if lease.live and HOST in lease.pools and "EB monitor" in lease.purpose]

    with memory_scope(_budget()) as broker:
        codes = EvaluationCodes(np.arange(10, dtype=np.int64), np.arange(4, dtype=np.int64), 4)
        assert purposes(broker) == ["the EB monitor's evaluation codes"]
        broker.evict_idle(HOST)
        assert codes.off is not None and codes.codes.shape[0] == 0 and not purposes(broker)
        assert capsys.readouterr().err.count("eb monitor: off: the ledger evicted") == 1
        codes = EvaluationCodes(np.arange(10, dtype=np.int64), np.arange(4, dtype=np.int64), 4)
        assert purposes(broker) == ["the EB monitor's evaluation codes"]
        del codes
        gc.collect()
        assert not purposes(broker)


def test_a_model_without_evaluation_rows_is_logged_unscored(capsys: pytest.CaptureFixture[str]) -> None:
    """A group's all-rows model holds no row out: its calls are logged with their ELBO and move, unscored, while the
    fold model beside it is scored."""
    from sv_pgs.scale_mixture_ep import MixtureHyperparameters

    codes = EvaluationCodes(np.arange(6, dtype=np.int64), np.arange(30, dtype=np.int64), 4)
    generator = np.random.default_rng(2)
    for start in (0, 4):
        codes.capture(np.arange(start, min(start + 4, 6), dtype=np.int64), generator.integers(0, 255, size=(min(4, 6 - start), 40), dtype=np.uint8))
    targets = np.column_stack([generator.standard_normal(30), np.full(30, np.nan)])
    evaluation = GroupEvaluation(np.arange(30, dtype=np.int64), np.ones((30, 1)), targets, np.array([False, False]), ("trait/fold0", "trait/all"))
    monitor = EbMonitor.start(codes, np.arange(6, dtype=np.int64), np.zeros(6), np.ones(6), evaluation)
    oracle = SimpleNamespace(mean=generator.standard_normal((6, 2)), elbo=np.array([-3.0, -4.0]))
    monitor.view([0, 1]).observe(oracle, [MixtureHyperparameters(coefficients=np.zeros(2), log_smoothing=np.zeros(0))] * 2, [0, 1])
    err = capsys.readouterr().err
    assert re.search(r"model trait/fold0 iter 1 elbo -3 pgs_partial_r2 \S+ x_step -", err)
    assert "model trait/all iter 1 elbo -4 not scored (no evaluation rows) x_step -" in err
    # A group of all-rows models alone has no rows and no codes, and is still logged.
    rows = EvaluationRows(np.arange(30, dtype=np.int64), np.zeros((30, 0)), targets, np.array([False, False]))
    alone = rows.group(np.array([1]), np.zeros(0, dtype=bool), ["trait/all"])
    assert alone.store_columns.shape == (0,) and alone.targets.shape == (0, 1)
    monitor = EbMonitor.start(None, np.arange(6, dtype=np.int64), np.zeros(6), np.ones(6), alone)
    monitor.view([0]).observe(SimpleNamespace(mean=oracle.mean[:, 1:], elbo=np.array([-4.0])), [MixtureHyperparameters(coefficients=np.zeros(2), log_smoothing=np.zeros(0))], [0])
    err = capsys.readouterr().err
    assert "no model of the group holds a row out" in err and "model trait/all iter 1 elbo -4 not scored (no evaluation rows)" in err


def test_a_member_stage0_never_read_leaves_every_call_unscored(capsys: pytest.CaptureFixture[str]) -> None:
    """Codes no Stage 0 tile filled (a resumed Stage 0) are never scored as zeros: the monitor says why, releases them,
    and logs each call's ELBO and move without metrics."""
    from sv_pgs.scale_mixture_ep import MixtureHyperparameters

    codes = EvaluationCodes(np.arange(10, dtype=np.int64), np.arange(4, dtype=np.int64), 4)
    codes.capture(np.arange(4, dtype=np.int64), np.zeros((4, 8), dtype=np.uint8))
    evaluation = GroupEvaluation(np.arange(4, dtype=np.int64), np.ones((4, 1)), np.zeros((4, 1)), np.array([False]), ("trait",))
    monitor = EbMonitor.start(codes, np.array([1, 6], dtype=np.int64), np.zeros(2), np.ones(2), evaluation)
    assert codes.off is not None and codes.codes.shape[0] == 0
    oracle = SimpleNamespace(mean=np.ones((2, 1)), elbo=np.array([-2.5]))
    monitor.view([0]).observe(oracle, [MixtureHyperparameters(coefficients=np.zeros(2), log_smoothing=np.zeros(0))], [0])
    err = capsys.readouterr().err
    assert "eb monitor: unscored" in err and "model trait iter 1 elbo -2.5 not scored (no evaluation codes) x_step -" in err


def test_a_tile_the_codes_cannot_take_turns_them_off_and_never_fails_stage0(capsys: pytest.CaptureFixture[str]) -> None:
    """A tile whose rows are not the codes' candidates turns the codes off; the read itself goes on."""
    codes = EvaluationCodes(np.arange(10, dtype=np.int64), np.arange(4, dtype=np.int64), 4)
    codes.capture(np.array([3, 7], dtype=np.int64), np.zeros((2, 8), dtype=np.uint8))
    assert codes.off is not None and "eb monitor: off" in capsys.readouterr().err


def _saved(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """A saved model's metadata (less its arrays file's hash: the zip archive stamps each entry's write time) and
    arrays."""
    from sv_pgs.artifact import _ARRAYS, _METADATA

    metadata = json.loads((path / _METADATA).read_text())
    metadata.pop("arrays_sha256")
    with np.load(path / _ARRAYS) as archive:
        return metadata, {name: archive[name] for name in archive.files}


def _checkpoints(work_dir: Path) -> dict[str, tuple[dict, dict[str, np.ndarray]]]:
    """Each checkpoint under ``work_dir`` by its path there (its key's directory and model), with its metadata (less
    the fit's elapsed seconds) and arrays."""
    found = {}
    for path in sorted((work_dir / "checkpoints").rglob("*.npz")):
        with np.load(path) as archive:
            metadata = json.loads(str(archive["metadata"]))
            metadata.pop("elapsed_seconds")
            found[path.relative_to(work_dir).as_posix()] = (metadata, {name: archive[name] for name in archive.files if name != "metadata"})
    return found


def _same(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    assert sorted(left) == sorted(right)
    for name in left:
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)


@pytest.mark.parametrize("route", ["samples", "gram"])
def test_a_monitored_fit_is_the_fit_without_the_monitor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], route: str) -> None:
    """Requested with its held-out rows as ``FitRequest.evaluation``, a fit is bit for bit the fit requested without
    them: every oracle call's ELBO, noise and hyperparameters (the EB trajectory), its checkpoint (key, path and
    content) and its saved model (arrays and metadata, the provenance's digests included), so the held-out targets
    reach nothing the fit computes, keys or keeps. And the monitor scored the fit's calls from the codes Stage 0's pass
    kept: each score is the evaluation rows' standardized store codes times the call's posterior mean."""
    from sv_pgs import fit_model, full_data_fit, stage2_wiring
    from sv_pgs.artifact import save_model

    store, covariate, targets, _genetic = _store(tmp_path / "store", 7)
    monkeypatch.setattr(stage2_wiring, "pass_costs", (lambda band, source, xp: (0.0, np.inf)) if route == "samples" else (lambda band, source, xp: (np.inf, 0.0)))
    # The request's draw count, cut to the engine tests' to keep the two fits short; both fits and keys see the same.
    monkeypatch.setattr(fit_model, "DRAW_COUNT", 16)
    monkeypatch.delenv(eb_monitor.MONITOR_SWITCH, raising=False)
    samples = store.n_samples
    held_out = np.arange(samples) >= samples * 4 // 5
    signed = store.read_codes(0, store.n_variants).astype(np.float64) - 127.0
    checked: list[float] = []
    scores = EbMonitor.scores

    def recorded(self, means):
        values = scores(self, means)
        members = self._codes.candidates[self._positions]
        direct = ((signed[members][:, self._codes.columns] - self._means[:, None]) / self._scales[:, None]).T @ means
        np.testing.assert_allclose(values, direct, rtol=1e-5, atol=1e-5 * float(np.max(np.abs(direct))))
        checked.append(float(np.max(np.abs(direct))))
        return values

    monkeypatch.setattr(EbMonitor, "scores", recorded)
    trajectory: list[list[np.ndarray]] = []
    call = full_data_fit._FullDataMeanField.__call__

    def traced(self, hyperparameters):
        points = call(self, hyperparameters)
        trajectory.append([np.array(self.elbo, copy=True), np.array(self.noise, copy=True)] + [
            np.array(values, copy=True) for hyper in hyperparameters for values in (hyper.coefficients, hyper.log_smoothing)
        ])
        return points

    monkeypatch.setattr(full_data_fit._FullDataMeanField, "__call__", traced)

    def fitted(name: str, evaluation: EvaluationRows | None):
        work_dir = tmp_path / name
        work_dir.mkdir()
        trajectory.clear()
        model = fit_model.fit(fit_model.FitRequest(
            store=store, store_columns=np.arange(samples, dtype=np.int64), covariates=covariate[:, None], covariate_names=("c",),
            covariate_columns=np.ones((1, 1), dtype=bool), targets=np.where(~held_out, targets, np.nan)[:, None], training=~held_out[:, None],
            model_names=("trait/fold4",), trait_types=(TraitType.QUANTITATIVE,), research_ids=tuple(f"person{index}" for index in range(samples)),
            log_variance_offset=None, budget=_budget(), work_dir=work_dir, seed=5, evaluation=evaluation,
        ))
        save_model(tmp_path / f"{name}_model", model)
        return list(trajectory), _checkpoints(work_dir), _saved(tmp_path / f"{name}_model")

    plain = fitted("plain", None)
    assert not checked
    capsys.readouterr()
    evaluation = EvaluationRows(
        store_columns=np.flatnonzero(held_out), covariates=covariate[held_out, None], targets=targets[held_out, None], disease=np.array([False]),
    )
    watched = fitted("watched", evaluation)
    log = capsys.readouterr().err
    assert checked and max(checked) > 0.0
    lines = [line for line in log.splitlines() if "eb monitor: model trait/fold4 iter" in line]
    assert len(lines) >= 2 and any("pgs_partial_r2" in line for line in lines)
    assert all(re.search(r"elbo -?\d", line) and "x_step" in line for line in lines)
    # Every line of the monitor's is one of its aggregate forms: metrics, ELBO and step norm, the time it took, or the
    # models it watches; no row, count or case.
    forms = (
        r"model \S+ iter \d+ elbo \S+ (pgs_partial_r2 \S+( pgs_auc \S+)?|not scored \([^)]*\)) x_step \S+ stage \w+",
        r"scored in [\d.]+ s \(\d+ s of the fit's \d+ s so far\)",
        r"on for trait/fold4: each outer-loop call logged, scored from the codes Stage 0's pass kept while the scores take at most 5% of the fit's time",
    )
    for line in log.splitlines():
        if "eb monitor:" in line:
            message = line.split("] ", 1)[1]
            assert any(re.fullmatch(f"eb monitor: {form}", message) for form in forms), message
    # The EB trajectory, call by call (the mode mixture's oracles' calls among them).
    assert len(watched[0]) == len(plain[0]) >= 2
    for watched_call, plain_call in zip(watched[0], plain[0]):
        assert len(watched_call) == len(plain_call)
        for left, right in zip(watched_call, plain_call):
            np.testing.assert_array_equal(left, right)
    # The checkpoint: the same key (its directory), the same content.
    assert plain[1] and sorted(watched[1]) == sorted(plain[1])
    for path, (metadata, arrays) in plain[1].items():
        assert watched[1][path][0] == metadata
        _same(watched[1][path][1], arrays)
    # The saved model.
    assert watched[2][0] == plain[2][0]
    _same(watched[2][1], plain[2][1])


def test_the_switch_turns_the_monitor_off_before_any_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """With MONITOR_SWITCH at 0 the evaluation rows reach no group's fit (no codes, no lease, no monitor line); with it
    unset they do."""
    from sv_pgs import stage2_wiring
    from sv_pgs.stage2_wiring import fit_models

    store, covariate, targets, _genetic = _store(tmp_path / "store", 7)
    samples = store.n_samples
    held_out = np.arange(samples) >= samples * 4 // 5
    seen: list[GroupEvaluation | None] = []

    class Stop(Exception):
        pass

    def group_fit(*arguments):
        seen.append(arguments[-1])
        raise Stop

    monkeypatch.setattr(stage2_wiring, "_fit_group", group_fit)
    evaluation = EvaluationRows(
        store_columns=np.flatnonzero(held_out), covariates=np.column_stack([np.ones(int(held_out.sum())), covariate[held_out]]),
        targets=targets[held_out, None], disease=np.array([False]),
    )
    for switch in ("0", None):
        if switch is None:
            monkeypatch.delenv(eb_monitor.MONITOR_SWITCH)
        else:
            monkeypatch.setenv(eb_monitor.MONITOR_SWITCH, switch)
        work_dir = tmp_path / f"work{len(seen)}"
        work_dir.mkdir()
        with pytest.raises(Stop):
            fit_models(
                store=store, store_columns=np.arange(samples, dtype=np.int64), covariates=np.column_stack([np.ones(samples), covariate]),
                covariate_columns=np.ones((1, 2), dtype=bool), targets=np.where(~held_out, targets, 0.0)[:, None], training=~held_out[:, None],
                trait_types=(TraitType.QUANTITATIVE,), log_variance_offset=None, budget=_budget(), work_dir=work_dir, seed=5, draw_count=16,
                evaluation=evaluation, model_names=("trait/fold4",),
            )
    first, second = seen
    assert first is None and isinstance(second, GroupEvaluation) and second.names == ("trait/fold4",)
    err = capsys.readouterr().err
    assert err.count(f"eb monitor: off ({eb_monitor.MONITOR_SWITCH}=0)") == 1


def test_the_fit_request_gives_the_monitor_each_model_s_held_out_fold(tmp_path: Path) -> None:
    """``workspace_pipeline.fit_request``: the evaluation rows are the held-out folds' rows with a finite target and
    finite covariates (the intercept left out, as the request's), each model's target NaN off its own fold, and a
    disease flagged whatever likelihood fits it; the training side of the request is unchanged by them."""
    from sv_pgs.workspace_pipeline import fit_request

    rows, generator = 40, np.random.default_rng(6)
    fold = np.arange(rows) % 2
    model_traits = np.array([0, 0, 0, 1, 1, 1])
    model_folds = np.array([0, 1, 2, 0, 1, 2])
    targets = np.column_stack([generator.standard_normal(rows), (generator.random(rows) < 0.4).astype(np.float64)])
    targets[5, 1] = np.nan
    covariates = np.column_stack([np.ones(rows), generator.standard_normal(rows)])
    held_out = (fold[:, None] == model_folds[None, :]) & np.isfinite(targets[:, model_traits])
    training = (fold[:, None] != model_folds[None, :]) & np.isfinite(targets[:, model_traits])
    # A row no model trains on (a development calibration row) whose covariate is missing: never an evaluation row.
    training[2], covariates[2, 1] = False, np.nan
    cohort = SimpleNamespace(
        store_columns=np.arange(rows, dtype=np.int64)[::-1].copy(), covariates=covariates, covariate_names=("intercept", "age"),
        covariate_columns=np.ones((2, 2), dtype=bool), targets=targets, training=training, held_out=held_out,
        model_names=("ldl/fold0", "ldl/fold1", "ldl/all", "t2d/fold0", "t2d/fold1", "t2d/all"), model_traits=model_traits,
        trait_types=(TraitType.QUANTITATIVE, TraitType.BINARY), research_ids=[f"person{row}" for row in range(rows)], fit_rows=training.any(axis=1),
    )
    request = fit_request(cohort, SimpleNamespace(n_samples=rows), None, _budget(), tmp_path, 3)
    evaluation = request.evaluation
    assert evaluation is not None
    kept = np.flatnonzero(held_out.any(axis=1) & (np.arange(rows) != 2))
    np.testing.assert_array_equal(evaluation.store_columns, cohort.store_columns[kept])
    np.testing.assert_array_equal(evaluation.covariates, covariates[kept, 1:])
    expected = np.where(held_out, targets[:, model_traits], np.nan)[kept]
    np.testing.assert_array_equal(evaluation.targets, expected)
    assert np.isnan(evaluation.targets[:, [2, 5]]).all() and evaluation.disease.tolist() == [False] * 3 + [True] * 3
    np.testing.assert_array_equal(request.training, training[training.any(axis=1)])
