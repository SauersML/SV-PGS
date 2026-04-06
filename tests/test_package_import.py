from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def test_import_sv_pgs_exports_symbols_directly():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import sv_pgs; "
                "print(json.dumps({"
                "'sv_pgs.all_of_us': 'sv_pgs.all_of_us' in sys.modules, "
                "'sv_pgs.benchmark': 'sv_pgs.benchmark' in sys.modules, "
                "'sv_pgs.io': 'sv_pgs.io' in sys.modules, "
                "'sv_pgs.model': 'sv_pgs.model' in sys.modules, "
                "'BayesianPGS': hasattr(sv_pgs, 'BayesianPGS'), "
                "'run_training_pipeline': hasattr(sv_pgs, 'run_training_pipeline'), "
                "'AllOfUsDiseaseRequest': hasattr(sv_pgs, 'AllOfUsDiseaseRequest')"
                "}))"
            ),
        ],
        capture_output=True,
        check=True,
        text=True,
    )

    loaded_modules = json.loads(completed.stdout.strip())
    assert loaded_modules == {
        "sv_pgs.all_of_us": False,
        "sv_pgs.benchmark": True,
        "sv_pgs.io": True,
        "sv_pgs.model": True,
        "BayesianPGS": True,
        "run_training_pipeline": True,
        "AllOfUsDiseaseRequest": False,
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
                    "BayesianPGS": hasattr(sv_pgs, "BayesianPGS"),
                    "AllOfUsDiseaseRequest": hasattr(sv_pgs, "AllOfUsDiseaseRequest"),
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
        "BayesianPGS": True,
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
