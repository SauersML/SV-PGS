"""The ops bucket's run listing (launcher/workspace/mirror_sizes.py), with no cloud calls: it sizes and never deletes."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from launcher.workspace import mirror_sizes
from sv_pgs import run_mirror
from sv_pgs.workspace_pipeline import COMPLETE_FILE, STEP_NAMES


def _object(name: str, size: int, crc: str, date: str) -> dict:
    return {"name": f"svpgs/{name}", "size": size, "crc32c_hash": crc, "update_time": f"{date}T12:00:00+00:00"}


def _whole_genome_run(run: str, date: str, final_crc: str = "final==") -> list[dict]:
    """A run whose store and direct_sv steps are complete, each with its store payload, its fit partway, and its
    staged environment and launcher logs."""
    listed = [
        (f"{run}/run/store/{COMPLETE_FILE}", 1, "a=="),
        (f"{run}/run/store/store/dosage/half0/chr1/c/0", 400 * 10**9, "store=="),
        (f"{run}/run/store.directories.json", 1, "b=="),
        (f"{run}/run/direct_sv/{COMPLETE_FILE}", 1, "c=="),
        (f"{run}/run/direct_sv/store/dosage/half0/chr1/c/0", 500 * 10**9, final_crc),
        (f"{run}/run/direct_sv/measurement_model.npz", 10**6, "d=="),
        (f"{run}/run/fit.partial/_KEY", 1, "e=="),
        (f"{run}/run/fit.partial/work/with_sv/checkpoints/k/model0.npz", 2 * 10**9, "f=="),
        (f"{run}/env.tar", 5 * 10**9, "g=="),
        (f"{run}/source/pyproject.toml", 10**3, "h=="),
        (f"{run}/launcher/train-1/batch.log", 10**6, "i=="),
        (f"{run}/budget/plan.json", 10**2, "j=="),
    ]
    return [_object(name, size, crc, date) for name, size, crc in listed]


def test_the_script_s_layout_is_the_pipeline_s_and_the_mirror_s() -> None:
    assert mirror_sizes.STEP_NAMES == STEP_NAMES and mirror_sizes.COMPLETE_FILE == COMPLETE_FILE
    assert mirror_sizes.PARTIAL == run_mirror.PARTIAL and mirror_sizes.STORE == run_mirror.STORE
    assert mirror_sizes.BASES == run_mirror.BASES and run_mirror.CARRIED == STEP_NAMES[: STEP_NAMES.index("fit")]
    assert run_mirror.Base(run_mirror.DirectoryRemote(Path(".")), "gs://ops/svpgs/base/run").reference == f"{run_mirror.BASES}/ops/svpgs/base/run"


def test_each_run_is_sized_by_part_and_a_store_several_runs_hold_is_named_once() -> None:
    objects = _whole_genome_run("hypertension", "2026-10-19") + _whole_genome_run("asthma", "2026-09-20")
    # A run whose direct_sv store differs, and one with only a store step partway: neither shares a payload.
    objects += _whole_genome_run("gout", "2026-10-19", final_crc="other==")
    objects += [_object("ldl/run/store.partial/_KEY", 1, "k==", "2026-10-18"), _object("ldl/run/store.partial/store/dosage/half0/chr1/c/0", 7 * 10**9, "l==", "2026-10-18")]
    text = mirror_sizes.report(objects, 0.02, datetime.date(2026, 10, 20))
    rows = {line.split()[0]: line.split() for line in text.splitlines()[1:5]}
    # idle days, TB, $/month, then restored, linked stores, partial, staged, logs, and the completed steps
    assert rows["hypertension"][1] == "1" and rows["asthma"][1] == "30"
    assert rows["hypertension"][2] == "0.907" and rows["hypertension"][4:9] == ["0.500", "0.400", "0.002", "0.005", "0.000"]
    assert rows["hypertension"][9] == "store,direct_sv"
    assert float(rows["hypertension"][3]) == round(sum(entry["size"] for entry in objects if entry["name"].startswith("svpgs/hypertension/")) / 2**30 * 0.02, 2)
    assert rows["ldl"][4:9] == ["0.000", "0.000", "0.007", "0.000", "0.000"] and rows["ldl"][9] == "-"
    assert "all runs" in text
    shared = [line for line in text.splitlines() if "held" in line]
    # The store step's payload is alike in all three runs, the direct_sv one in two of them.
    assert len(shared) == 2 and shared[0].strip().startswith("0.400 TB held 3 times, 0.800 TB beyond one copy")
    assert "0.500 TB held 2 times, 0.500 TB beyond one copy" in shared[1]
    assert "asthma/run/direct_sv/store" in shared[1] and "hypertension/run/direct_sv/store" in shared[1] and "gout" not in shared[1]


def test_a_base_mirror_is_named_with_the_runs_it_carries_and_flagged_once_it_holds_no_completed_step() -> None:
    """A run whose mirror a base carries steps for names it under run/.bases/ (sv_pgs.run_mirror.Base.reference)."""
    named = lambda run, base: _object(f"{run}/run/.bases/workspace/svpgs/{base}/run", 0, "AAAAAA==", "2026-10-19")
    objects = _whole_genome_run("hypertension", "2026-10-19") + [_object("ldl/run/fit.partial/_KEY", 1, "k==", "2026-10-19")]
    objects += [named("ldl", "hypertension"), named("gout", "hypertension"), named("asthma", "pruned")]
    # gout still holds a copy of the base's store step object for object (a server-side copy), and its own direct_sv.
    objects += [entry for entry in _whole_genome_run("gout", "2026-10-19", final_crc="other==") if "/run/store/" in entry["name"] or "/run/direct_sv/" in entry["name"]]
    text = mirror_sizes.report(objects, 0.02, datetime.date(2026, 10, 20))
    assert text.split("base mirrors that carry other runs' completed steps (keep each while those runs are kept):\n")[1].splitlines()[:2] == [
        "  hypertension: carries gout, ldl",
        "  pruned: carries asthma  HOLDS NO COMPLETED STEP: their next restores make those steps again",
    ]
    (removable,) = [line for line in text.splitlines() if line.startswith("  gout/run/")]
    assert removable == f"  gout/run/store/ as hypertension/run/store/: 0.400 TB, ${(400 * 10**9 + 1) / 2**30 * 0.02:.2f}/month"


def test_the_bucket_is_only_listed(tmp_path: Path) -> None:
    """Against a local fake gcloud that answers a listing and refuses every other command."""
    executable = tmp_path / "gcloud"
    executable.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
Path(os.environ["CALLS"]).open("a").write(json.dumps(args) + "\\n")
if args[:3] != ["storage", "objects", "list"]:
    sys.exit(f"refused: {args}")
print(Path(os.environ["LISTING"]).read_text())
''')
    executable.chmod(0o755)
    listing = tmp_path / "objects.json"
    listing.write_text(json.dumps(_whole_genome_run("hypertension", "2026-10-19")))
    environment = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"], CALLS=str(tmp_path / "calls"), LISTING=str(listing))
    script = Path(__file__).parents[1] / "launcher/workspace/mirror_sizes.py"
    shown = subprocess.run([sys.executable, str(script), "--bucket", "workspace", "--usd-per-gib-month", "0.02"], env=environment, capture_output=True, text=True)
    assert shown.returncode == 0, shown.stderr
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    assert calls == [["storage", "objects", "list", "gs://workspace/svpgs/**", "--format=json(name,size,crc32c_hash,update_time)"]]
    assert "hypertension" in shown.stdout
    saved = subprocess.run([sys.executable, str(script), "--listing", str(listing), "--usd-per-gib-month", "0.02"], capture_output=True, text=True)
    assert saved.returncode == 0 and saved.stdout.splitlines()[1].split()[2:] == shown.stdout.splitlines()[1].split()[2:]
