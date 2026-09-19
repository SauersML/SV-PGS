from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


KEPT_MODULES = (
    "sv_pgs.code_products",
    "sv_pgs.compute_budget",
    "sv_pgs.dosage_store",
    "sv_pgs.exact_polish",
    "sv_pgs.fast_scoring",
    "sv_pgs.genotype_buffers",
    "sv_pgs.genotype_statistics",
    "sv_pgs.store_converter",
    "sv_pgs.synthetic_store",
)


def test_kept_modules_load_neither_jax_nor_the_old_path():
    """Importing the package or any kept module must not import JAX or change the CUDA math mode.

    The old path's ``_jax`` shim imported JAX, enabled x64 and set ``CUPY_TF32`` for the whole
    process as an import side effect.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import importlib, json, os, sys
                import sv_pgs
                for name in {KEPT_MODULES!r}:
                    importlib.import_module(name)
                print(json.dumps({{
                    "jax": sorted(name for name in sys.modules if name.split(".")[0] in ("jax", "jaxlib")),
                    "old": sorted(
                        name for name in sys.modules
                        if name in ("sv_pgs._jax", "sv_pgs.genotype", "sv_pgs.io", "sv_pgs.model")
                    ),
                    "CUPY_TF32": os.environ.get("CUPY_TF32"),
                    "exports": sorted(sv_pgs.__all__),
                }}))
                """
            ),
        ],
        capture_output=True,
        check=True,
        env={key: value for key, value in os.environ.items() if key != "CUPY_TF32"},
        text=True,
    )

    loaded = json.loads(completed.stdout.strip())
    assert loaded == {
        "jax": [],
        "old": [],
        "CUPY_TF32": None,
        "exports": ["ModelConfig", "TraitType", "VariantClass", "VariantRecord"],
    }


def test_import_sv_pgs_succeeds_without_bigquery():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import builtins
                import json

                real_import = builtins.__import__

                def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
                    if name == "google.cloud" and "bigquery" in fromlist:
                        raise ModuleNotFoundError("No module named 'google.cloud.bigquery'")
                    return real_import(name, globals, locals, fromlist, level)

                builtins.__import__ = blocked_import

                import sv_pgs

                print(json.dumps({
                    "ModelConfig": hasattr(sv_pgs, "ModelConfig"),
                    "sv_pgs.all_of_us": "sv_pgs.all_of_us" in __import__("sys").modules,
                    "AllOfUsDiseaseRequest": "AllOfUsDiseaseRequest" in sv_pgs.__dict__,
                }))
                """
            ),
        ],
        capture_output=True,
        check=True,
        text=True,
    )

    loaded_symbols = json.loads(completed.stdout.strip())
    assert loaded_symbols == {
        "ModelConfig": True,
        "sv_pgs.all_of_us": False,
        "AllOfUsDiseaseRequest": False,
    }


def test_repo_root_does_not_shadow_installed_cyvcf2():
    repo_root = Path(__file__).resolve().parents[1]
    shadow_path = repo_root / "cyvcf2.py"
    pytest.importorskip("cyvcf2")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib, cyvcf2; print(pathlib.Path(cyvcf2.__file__).resolve())",
        ],
        capture_output=True,
        check=True,
        cwd=repo_root,
        text=True,
    )

    imported_path = Path(completed.stdout.strip())
    assert imported_path != shadow_path
