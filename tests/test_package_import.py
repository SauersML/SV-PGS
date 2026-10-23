from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap


def test_importing_the_package_loads_no_jax_and_keeps_the_cuda_math_mode():
    """Importing the package or any of its modules must not import JAX or change the CUDA math mode."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import importlib, json, os, pkgutil, sys
                import sv_pgs
                for module in pkgutil.walk_packages(sv_pgs.__path__, "sv_pgs."):
                    importlib.import_module(module.name)
                print(json.dumps({
                    "jax": sorted(name for name in sys.modules if name.split(".")[0] in ("jax", "jaxlib")),
                    "CUPY_TF32": os.environ.get("CUPY_TF32"),
                    "exports": sorted(sv_pgs.__all__),
                }))
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
        "CUPY_TF32": None,
        "exports": ["ModelConfig", "TraitType", "VariantClass"],
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
    }

