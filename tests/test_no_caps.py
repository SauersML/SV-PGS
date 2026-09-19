"""SPEC.md: the number of variants or samples is never capped.

A cap that comes back is a SPEC violation: delete the cap, not the test.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import sv_pgs
from sv_pgs.config import ModelConfig

CAP_FIELDS = ("maximum_active_variants", "max_variants", "maximum_samples", "max_samples", "screen_max", "variant_budget")
CAP_NAMES = ("maximum_active_variants", "maximum_samples", "max_samples", "screen_max", "variant_budget")


def test_model_config_has_no_cap_field() -> None:
    fields = [field.name for field in dataclasses.fields(ModelConfig)]
    assert [field for field in fields if any(cap in field for cap in CAP_FIELDS)] == []


def test_no_module_names_a_cap() -> None:
    offenders = [
        (path.name, cap)
        for path in Path(sv_pgs.__file__).parent.rglob("*.py")
        for cap in CAP_NAMES
        if cap in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
