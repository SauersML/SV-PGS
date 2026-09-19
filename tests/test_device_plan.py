"""The device plan: LPT block assignment, the exposed-device check, and the reassociation bound."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.device_plan import check_visible_devices, exposed_device_count, plan_blocks, reassociation_bound


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
