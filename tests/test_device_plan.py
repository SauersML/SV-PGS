"""The device plan: LPT block assignment, the exposed-device check, and the reassociation bound."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.compute_budget import ComputeBudget, check_visible_devices, exposed_device_count
from sv_pgs.device_plan import plan_blocks, reassociation_bound


def _budget(device_ids: tuple[int, ...]) -> ComputeBudget:
    if not device_ids:
        return ComputeBudget(device_kind="cpu", device_ids=(), device_names=(), device_bytes=(),
                             device_compute_capabilities=(), host_bytes=1 << 30, cpu_threads=1)
    return ComputeBudget(device_kind="cuda", device_ids=device_ids, device_names=tuple("A40" for _ in device_ids),
                         device_bytes=tuple(1 << 30 for _ in device_ids), device_compute_capabilities=tuple((8, 6) for _ in device_ids),
                         host_bytes=1 << 30, cpu_threads=1)


def test_blocks_go_widest_first_to_the_least_loaded_device() -> None:
    plan = plan_blocks((0, 1), [5, 9, 4, 3], 10)
    # 9 -> device 0; 5 -> device 1; 4 -> device 1 (load 5 < 9); 3 -> device 0 (loads tie at 9, lower position)
    np.testing.assert_array_equal(plan.block_devices, [1, 0, 1, 0])
    assert plan.loads == ((9 + 3) * 10, (5 + 4) * 10)
    np.testing.assert_array_equal(plan.blocks_on(0), [1, 3])
    np.testing.assert_array_equal(plan.blocks_on(1), [0, 2])
    assert "2 device(s)" in plan.describe()


def test_one_device_takes_every_block_in_order() -> None:
    plan = plan_blocks((3,), [4, 4, 7], 5)
    np.testing.assert_array_equal(plan.block_devices, [0, 0, 0])
    np.testing.assert_array_equal(plan.blocks_on(0), [0, 1, 2])


def test_the_plan_refuses_empty_devices_and_empty_blocks() -> None:
    with pytest.raises(ValueError):
        plan_blocks((), [3], 1)
    with pytest.raises(ValueError):
        plan_blocks((0,), [3, 0], 1)


@pytest.mark.parametrize(
    ("visible", "device_ids", "raises"),
    [("0,1", (0, 1), False), ("0,1", (0,), True), ("0", (), True), ("", (), False), ("-1", (), False)],
)
def test_every_exposed_device_must_be_usable(monkeypatch, visible, device_ids, raises) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert exposed_device_count() == len([entry for entry in visible.split(",") if entry not in ("", "-1")])
    if raises:
        with pytest.raises(RuntimeError, match="exposed"):
            check_visible_devices(_budget(device_ids))
    else:
        check_visible_devices(_budget(device_ids))


def test_two_summation_orders_differ_within_the_reassociation_bound() -> None:
    rng = np.random.default_rng(3)
    images = [rng.standard_normal((50, 4)) * 10.0 ** rng.integers(-8, 8) for _ in range(33)]
    forward = np.zeros((50, 4))
    for image in images:
        forward += image
    by_halves = [np.zeros((50, 4)), np.zeros((50, 4))]
    for index, image in enumerate(images):
        by_halves[index % 2] += image
    assert np.all(np.abs(forward - (by_halves[0] + by_halves[1])) <= reassociation_bound(images, np))


def test_stage2_shards_its_reads_over_the_devices_as_one_streamed_source_reads(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 2's reads are sharded over a budget's CUDA devices by LD block (``stage2_wiring.stage2_source``), on a model's
    training samples. Checked on the host: numpy stands in for CuPy (``compute_budget._cupy_device_context`` needs no
    device there) and each device's shard is a host source, so the plan, the sample selection, the peer tiles and the
    threads of ``map_reduce`` run as on two devices. Every block's products are the streamed source's bit for bit, a
    read summed over the devices is within the reassociation bound of the streamed one, and a mean-field fit is the
    streamed source's to its tolerance."""
    from sv_pgs import compute_budget, dual_solve, stage2_wiring, store_block_source
    from sv_pgs.config import ModelConfig
    from sv_pgs.device_plan import ShardedDualSource
    from sv_pgs.full_data_fit import block_grams, covariate_residual_variance, fit_full_data, stage0_lattice
    from sv_pgs.genotype_statistics import DosageStoreTileSource, compute_genotype_statistics
    from sv_pgs.scale_mixture_ep import scale_mixture_prior
    from tests.test_full_data_fit import _BLOCK_CAP, _DRAWS, _SAMPLES, _TRAINING, _WORKSPACE_BYTES, _store

    store, covariate, targets, _genetic = _store(tmp_path / "store", 7)
    training = np.arange(_TRAINING, dtype=np.int64)
    covariates = np.column_stack([np.ones(_TRAINING), covariate[training]])
    host = _budget(())
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(store.n_variants)), training, covariates, targets[training, None], ModelConfig(), host,
        _BLOCK_CAP, tmp_path / "ld",
    )
    host_source = store_block_source.StoreGenotypeBlockSource

    def host_shard(store, block_rows, block_indices, means, scales, _device_budget, workspace_bytes, sample_indices=None):
        return host_source(store, block_rows, block_indices, means, scales, host, workspace_bytes, sample_indices)

    monkeypatch.setattr(compute_budget, "_try_import_cupy", lambda: np)
    monkeypatch.setattr(store_block_source, "StoreGenotypeBlockSource", host_shard)
    streamed = stage2_wiring.stage2_source(store, statistics, host, _WORKSPACE_BYTES, training)
    sharded = stage2_wiring.stage2_source(store, statistics, _budget((0, 1)), _WORKSPACE_BYTES, training)
    assert isinstance(sharded, ShardedDualSource) and isinstance(streamed, dual_solve.StreamedDualSource)
    assert sharded.sample_count == streamed.sample_count == _TRAINING and sharded.variant_count == streamed.variant_count
    assert len(set(sharded.plan.block_devices.tolist())) == 2 and sharded.block_bounds == streamed.block_bounds

    generator = np.random.default_rng(5)
    right = generator.standard_normal((sharded.variant_count, 2))
    left = generator.standard_normal((_TRAINING, 2))
    images = []
    for (start, stop, single), (sharded_start, sharded_stop, peer) in zip(streamed.blocks(), sharded.blocks(), strict=True):
        assert (start, stop) == (sharded_start, sharded_stop)
        np.testing.assert_array_equal(peer.matmat(right[start:stop]), single.matmat(right[start:stop]))
        np.testing.assert_array_equal(peer.rmatmat(left), single.rmatmat(left))
        np.testing.assert_array_equal(peer.columns(np.arange(stop - start)), single.columns(np.arange(stop - start)))
        images.append(single.matmat(right[start:stop] * single.rmatmat(left)))

    def work(start, stop, tile, shared, rows, image):
        image += tile.matmat(rows["right"] * tile.rmatmat(shared["left"]))

    total = sharded.map_reduce(work, {"left": left}, {"right": right}, tuple(left.shape))
    sequential = streamed.map_reduce(work, {"left": left}, {"right": right}, tuple(left.shape))
    assert np.all(np.abs(total - sequential) <= reassociation_bound(images, np))

    # A mean-field fit on each source: the sharded one's reads differ from the streamed one's only by the order a read's
    # devices are summed in.
    mask = np.zeros((_SAMPLES, 1))
    mask[training, 0] = 1.0
    store_covariates = np.column_stack([np.ones(_SAMPLES), covariate])
    members = statistics.active_rows.shape[0]
    noise = float(covariate_residual_variance(targets[:, None], mask, store_covariates)[0])
    nodes, floor, top = stage0_lattice(statistics, 0, noise, np.zeros(members), 0.5 / _DRAWS)
    prior = scale_mixture_prior(
        class_index=np.zeros(members, dtype=np.int64), log_variance_offset=np.zeros(members), annotation_design=np.zeros((members, 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )
    means = []
    for source in (streamed, sharded):
        gaussian = dual_solve.DualGaussian(
            source=source, training=np.ones((_TRAINING, 1)), targets=targets[training, None], offsets=np.zeros((_TRAINING, 1)),
            covariates=covariates, grams=block_grams(statistics), probe_count=_DRAWS, seed=11, probe_sample_indices=training,
        )
        fit = fit_full_data(gaussian=gaussian, statistics=statistics, prior=prior, draw_count=_DRAWS, working_bytes=1 << 22, seed=13, inference="mean_field")
        means.append(fit.member_mean[:, 0])
    assert np.corrcoef(*means)[0, 1] > 0.999
