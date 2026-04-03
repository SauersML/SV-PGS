from __future__ import annotations

import json
import subprocess
import sys


def test_import_sv_pgs_does_not_eagerly_load_heavy_submodules():
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
                "'sv_pgs.model': 'sv_pgs.model' in sys.modules"
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
        "sv_pgs.benchmark": False,
        "sv_pgs.io": False,
        "sv_pgs.model": False,
    }
