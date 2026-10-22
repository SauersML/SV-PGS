"""List and size the runs kept in the workspace's ops bucket, so the user can decide which to delete. It deletes and
writes nothing: it reads one object listing, from gcloud or from a file.

Each run keeps under gs://<OPS_BUCKET>/svpgs/<RUN_ID>/ its staged environment (source/, env.tar, run_config.json), its
launcher logs, markers and budget ledger, and its mirror run/ (sv_pgs/run_mirror.py): each completed step <step>/ with
its COMPLETE_FILE, each step's durable partial work <step>.partial/, and each step's directory list. At the whole-genome
workload a mirror holds 0.5-1.5 TB, most of it copies of the genotype codes (<step>/store/): the store, measurement and
direct_sv steps may each hold one, and a restore downloads only the last; the others it links on the bucket mount, and
they are read only to make a later step again.

For each run it prints the bytes by part, the days since the run last wrote an object, and the monthly cost at the price
given; then each store payload that several runs hold byte for byte (every object's size and CRC32C alike), which one
shared copy could replace; then each base mirror that carries other runs' completed steps (their run/.bases/ names it),
which must be kept while they are, and any such base that holds nothing. Cloud Storage bills stored bytes by the
GiB-month at the bucket's location and class.

  python3 launcher/workspace/mirror_sizes.py --bucket "$OPS_BUCKET" --usd-per-gib-month 0.020
  gcloud storage objects list "gs://$OPS_BUCKET/svpgs/**" --format=json > objects.json
  python3 launcher/workspace/mirror_sizes.py --listing objects.json --usd-per-gib-month 0.020
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

STEP_NAMES = ("samples", "store", "phenotypes", "cohort", "measurement", "direct_sv", "fit", "score", "report", "effects", "export")
"""sv_pgs.workspace_pipeline.STEP_NAMES, in its order; the app's python3 that runs this script has no sv_pgs."""
COMPLETE_FILE = "_COMPLETE.json"
PARTIAL = ".partial"
STORE = "store"
BASES = ".bases"
CARRIED = STEP_NAMES[: STEP_NAMES.index("fit")]
"""sv_pgs.run_mirror.CARRIED: the steps a base mirror carries for the runs that name it, those before the fit."""
ROOT = "svpgs/"
STAGED = ("source", "env.tar", "run_config.json")
PARTS = ("restored", "linked stores", "partial", "staged", "logs")
"""A run's parts: the completed steps a restore downloads; the store payloads a later completed step's store replaces,
which a restore links on the bucket mount instead; the steps' durable partial work; the staged environment; the
launcher's logs, markers and budget ledger (keep the ledger: it is the study envelope's record)."""


def listed(bucket: str) -> list[dict]:
    """Every object under gs://<bucket>/svpgs/, with its size, CRC32C and update time."""
    shown = subprocess.run(
        ["gcloud", "storage", "objects", "list", f"gs://{bucket}/{ROOT}**", "--format=json(name,size,crc32c_hash,update_time)"],
        capture_output=True, check=True, text=True,
    )
    return json.loads(shown.stdout or "[]")


def runs_of(objects: list[dict]) -> dict[str, list[tuple[str, int, str, str]]]:
    """Each run's objects as (name below svpgs/<RUN_ID>/, bytes, CRC32C, update date)."""
    runs: dict[str, list[tuple[str, int, str, str]]] = defaultdict(list)
    for entry in objects:
        name = str(entry["name"])
        if not name.startswith(ROOT) or name.count("/") < 2:
            continue
        run, inside = name[len(ROOT):].split("/", 1)
        runs[run].append((inside, int(entry["size"]), str(entry.get("crc32c_hash", "")), str(entry.get("update_time", ""))[:10]))
    return runs


def parts_of(entries: list[tuple[str, int, str, str]]) -> tuple[dict[str, str], dict[str, list[tuple[str, int, str]]]]:
    """Each object's part (PARTS), and each completed step's store payload as (name below store/, bytes, CRC32C)."""
    names = {inside for inside, _size, _crc, _date in entries}
    complete = [step for step in STEP_NAMES if f"run/{step}/{COMPLETE_FILE}" in names]
    payloads: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for inside, size, crc, _date in entries:
        parts = inside.split("/")
        if parts[0] == "run" and len(parts) >= 4 and parts[1] in complete and parts[2] == STORE:
            payloads[parts[1]].append(("/".join(parts[3:]), size, crc))
    # As sv_pgs.run_mirror._replaced_stores: every completed step's store but the last one's.
    linked = [step for step in STEP_NAMES if step in payloads][:-1]
    part = {}
    for inside, _size, _crc, _date in entries:
        parts = inside.split("/")
        if parts[0] != "run":
            part[inside] = "staged" if parts[0] in STAGED else "logs"
        elif len(parts) >= 3 and parts[1].endswith(PARTIAL):
            part[inside] = "partial"
        elif len(parts) >= 4 and parts[1] in linked and parts[2] == STORE:
            part[inside] = "linked stores"
        else:
            part[inside] = "restored"
    return part, payloads


def report(objects: list[dict], usd_per_gib_month: float, today: datetime.date) -> str:
    runs = runs_of(objects)
    lines = [f"{'run':<28}{'idle days':>10}{'TB':>9}{'$/month':>10}" + "".join(f"{name + ' TB':>18}" for name in PARTS) + "  completed steps"]
    holders: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)
    for run in sorted(runs):
        entries = runs[run]
        part, payloads = parts_of(entries)
        sizes: dict[str, int] = defaultdict(int)
        for inside, size, _crc, _date in entries:
            sizes[part[inside]] += size
            totals[part[inside]] += size
        newest = max((date for *_rest, date in entries if date), default="")
        idle = str((today - datetime.date.fromisoformat(newest)).days) if newest else "?"
        held = sum(sizes.values())
        complete = [step for step in STEP_NAMES if f"run/{step}/{COMPLETE_FILE}" in part]
        lines.append(
            f"{run:<28}{idle:>10}{held / 1e12:>9.3f}{held / 2**30 * usd_per_gib_month:>10.2f}"
            + "".join(f"{sizes[name] / 1e12:>18.3f}" for name in PARTS) + "  " + (",".join(complete) or "-")
        )
        for step, files in payloads.items():
            signature = hashlib.sha256(json.dumps(sorted(files)).encode()).hexdigest()
            holders[signature].append((run, step, sum(size for _name, size, _crc in files)))
    held = sum(totals.values())
    lines.append(
        f"{'all runs':<28}{'':>10}{held / 1e12:>9.3f}{held / 2**30 * usd_per_gib_month:>10.2f}"
        + "".join(f"{totals[name] / 1e12:>18.3f}" for name in PARTS)
    )
    shared = sorted((group for group in holders.values() if len(group) > 1), key=lambda group: -group[0][2] * (len(group) - 1))
    # sv_pgs.run_mirror.Base.reference: run/.bases/<bucket>/<prefix>, svpgs/<base>/run for a run of this bucket.
    carried: dict[str, list[str]] = defaultdict(list)
    for run, entries in runs.items():
        for inside, _size, _crc, _date in entries:
            parts = inside.split("/")
            if parts[:2] == ["run", BASES] and len(parts) > 2:
                base = parts[4] if len(parts) == 6 and parts[3] == ROOT.strip("/") and parts[5] == "run" else "gs://" + "/".join(parts[2:])
                carried[base].append(run)
    if shared:
        lines.append("")
        lines.append("store payloads several runs hold byte for byte (one copy could serve them all):")
        for group in shared:
            spare = group[0][2] * (len(group) - 1)
            lines.append(
                f"  {group[0][2] / 1e12:.3f} TB held {len(group)} times, {spare / 1e12:.3f} TB beyond one copy "
                f"(${spare / 2**30 * usd_per_gib_month:.2f}/month): " + ", ".join(f"{run}/run/{step}/{STORE}" for run, step, _size in group)
            )
    if carried:
        lines.append("")
        lines.append("base mirrors that carry other runs' completed steps (keep each while those runs are kept):")
        redundant = []
        for base in sorted(carried):
            kept = {inside: (size, crc) for inside, size, crc, _date in runs.get(base, ())}
            gone = not base.startswith("gs://") and not any(f"run/{step}/{COMPLETE_FILE}" in kept for step in CARRIED)
            lines.append(
                f"  {base}: carries {', '.join(sorted(carried[base]))}"
                + ("  HOLDS NO COMPLETED STEP: their next restores make those steps again" if gone else "")
            )
            for run in sorted(carried[base]):
                for step in CARRIED:
                    own = {inside: (size, crc) for inside, size, crc, _date in runs[run] if inside.startswith(f"run/{step}/")}
                    if f"run/{step}/{COMPLETE_FILE}" in own and own == {inside: held for inside, held in kept.items() if inside.startswith(f"run/{step}/")}:
                        redundant.append((run, step, base, sum(size for size, _crc in own.values())))
        if redundant:
            lines.append("")
            lines.append(
                "completed steps a run holds as its base does, object for object (size and CRC32C): its jobs that read the "
                "base (BASE_RUN, run-mirror --base) restore them from there, so this copy can be deleted once no job of "
                "the run runs older code:"
            )
            for run, step, base, size in redundant:
                lines.append(f"  {run}/run/{step}/ as {base}/run/{step}/: {size / 1e12:.3f} TB, ${size / 2**30 * usd_per_gib_month:.2f}/month")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bucket", help="The ops bucket's name, without gs://; its svpgs/ prefix is listed with gcloud.")
    source.add_argument("--listing", type=Path, help="A saved `gcloud storage objects list --format=json` of that prefix.")
    parser.add_argument("--usd-per-gib-month", type=float, required=True, help="The bucket's storage price at its location and class.")
    args = parser.parse_args()
    objects = json.loads(args.listing.read_text()) if args.listing is not None else listed(args.bucket)
    print(report(objects, args.usd_per_gib_month, datetime.date.today()))


if __name__ == "__main__":
    main()
