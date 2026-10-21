"""One command for a round of the study's fits: every trait and arm, from one staged source to every job submitted, with
the plan printed first. Never launched by an agent: launching is the user's decision, and only --execute changes
anything (it stages, writes, copies and submits); without it the command only reads, and with --offline not even that.

    python3 launcher/workspace/round.py ROUND.json [--timeline budget|today|projected|TIMELINE.json] [--offline]
    python3 launcher/workspace/round.py ROUND.json --execute
    python3 launcher/workspace/round.py ROUND.json --follow [--execute] [--watch MINUTES]

A round was launched by hand-written scripts, run by run: each run's source copied on its own (one at a time, since
copies run at once left short trees), its completed steps copied, its config written and a job submitted per arm, with
a watcher merging each trait's two arms. This does the same from one spec (ROUND.json, below) and prints first what it
will do: the runs, the steps each takes from its base run and the ones its job makes, the jobs with each one's
environment, the minutes of each phase of a job on a timeline (the round's budget, today's, the owners' projections or
a measured one), the round's wall clock, its dollars (launcher/workspace/round_cost.py's projection against today's
layout where the module is there, else spend.py's prices times the minutes), and the checks it made and will make
before it submits anything. A check marked FAIL stops --execute before it changes anything.

What a round uses:
- One staged source, svpgs/_rounds/<round>/ in the ops bucket: the commit's tree (git archive), the digests of its
  files (source.sha256, which every job checks its own copy against before it runs anything, run_step.sh.template) and
  env.tar, copied from a run staged at the same uv.lock and pyproject.toml. Every job names it as STAGED_SOURCE, so no
  run has a source of its own and a round stages once. Each run's STAGED_SOURCE object names it, for a later round
  whose base the run is.
- The steps before the fit (samples .. direct_sv) from each trait's base run, where their keys hold: by default through
  BASE_RUN (run_mirror's bases: the job restores them from the base's mirror and its push names the base, so no copy is
  made and the base must be kept as long as the run is), or with "reuse": "copy" server-side into each run's own mirror,
  each step's objects first and its _COMPLETE.json last, once the copy's names and sizes are the base's. A step's key
  is its code, its upstream steps' keys and its inputs, the run config's (``workspace_pipeline._run_step``). A run's
  config is its base run's, but for its marker directory and the keys only the fit and later steps read
  (CONFIG_OVERRIDES), so the inputs hold; the code is compared between the base's staged source and the commit, by
  file where the trees' sv_pgs are the same and otherwise by the commit's launcher/workspace/step_codes.py; a step whose
  code differs, or that reads such a step, is the job's to make. Where the commit's driver has accept_steps_from, the
  config names the base's source, so a record made before the key's formula changed is still taken where that source
  made it from these inputs with this code.
- Each arm's empirical Bayes started from its own arm's saved model of the trait in the trait's warm_from runs (the
  live round's, by default its base run), where the commit's driver has warm_start_models (warm_seeds): the config
  names run/fit*/models/{arm}/{trait} of each such run, the same patterns for both arms of a trait so their fit keys
  agree. The plan reads each model's model.json (names and digests, no values) and says per arm whether it starts warm
  and from which model, or cold and why; a trait whose two arms would start unlike (one warm, one cold) fails the plan,
  since the comparison's difference must be the data's, not the starts'.
- Every job: SHARED_RUNTIME=1, Spot with MAX_RETRIES 3 under the job templates' RETRY_TASK lifecycle (a retry only when
  the VM went away), unless the spec says otherwise.
- A fit job's shape: one A100 on four local SSDs (ONE_GPU) unless the spec names another. No arm has run on one GPU
  yet, so "canaries" names the arms that do first, and every other arm takes TWO_GPU, the live rounds' machine; with no
  canaries named every arm takes the round's shape, and the plan warns while that is one GPU.

LAYOUTS: arm_jobs, one run per trait and arm (FIT_ARMS); trait_jobs, one run per trait whose one VM fits both arms at
once (PARALLEL_ARMS=1, one GPU each); arm_vms, one store job (submit.sh store) that makes the cohort of every trait and
its steps up to the fit once, then one run per arm whose VM fits every trait (ag-share's: cross_fit false, the compact
fit store and the arm pass, each where the commit's driver has it), its groups in turn until a fit calls
device_sweep's 11-trait sweep. A trait's two arms (arm_jobs), or the round's (arm_vms), come together in one of two
ways, by what the commit's task knows:
- with PARTNER_RUN (ag-score's), each arm's job scores its own arm and the later of the two finishes the run on its VM,
  minutes after its fit: no merge VM, restore or digest. A merge job on CPUs (submit.sh merge) is submitted by --follow
  only where both fit jobs have ended, neither run holds the round's last step and each holds its arm's handoff, in
  the run that got further;
- without it, as the hand-launched rounds did: once both fit jobs have SUCCEEDED with one fit key, --follow copies the
  other arm's models into one run's fit partial and submits that run on the merge slot (a merge job: its restore,
  digest, both arms' scoring and the report).
--follow also submits arm_vms' fit jobs once its store job has SUCCEEDED; with --watch it looks again every so many
minutes until nothing waits.

ROUND.json:
  {"round": "r7", "source_commit": "<sha>", "layout": "arm_jobs",
   "traits": [{"name": "ldl_cholesterol", "code": "l", "base_run": "aou-wl5"}, ...],
   "canaries": ["l:snv_indel", "l:with_sv"]}
  and optionally "arms", "run_prefix" (aou-<round>), the fit's "machine_type", "gpu_type", "gpu_count",
  "run_disk_type" and "run_disk_gb" (the layout's: ONE_GPU, TWO_GPU or ARM_VM; a named A2's GPUs from its name), "provisioning_model" (SPOT), "max_retries" (3 on Spot, else 0),
  "through" (report), "fit_slot" (train-1), "merge_slot" (train-2), "store_slot" (prepare), "hourly_usd" (the budget
  guard's rate; the priced rate rounded up), "env_from" (the first trait's base run), "reuse" (base or copy),
  "reuse_steps" (samples .. direct_sv), "config" (CONFIG_OVERRIDES' keys), "one_gpu_slowdown" (an arm's EB on one GPU
  against two under trait_jobs; 1.0), "merge_machine_type" (a merge job's without PARTNER_RUN; its run's),
  "merge_cpu_machine_type" (a merge job's with PARTNER_RUN; CPU_MERGE's), "store_machine_type" (arm_vms' store job's
  CPU machine; STORE_MACHINE's) and "warm_start" (true: each arm from its own earlier model; false: every arm
  cold). A trait may carry its own machine_type, gpu_type, gpu_count, run_disk_type and run_disk_gb, as a trait whose
  codes outgrow one A100-40 does, and its "warm_from", the runs whose saved models start its arms (its base_run; name
  both runs of a live pair, such as ["aou-wl5", "aou-wl1"], while their merge has not put both arms in one); a canary
  is TRAIT:ARM, by the trait's code or name.
The workspace's own settings come from the environment, as for submit.sh (WORKSPACE_ENVIRONMENT).

Standard library only, Python 3.8 or later (the workspace app's python3). It prints names, sizes, minutes and dollars,
never a value of the data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    from launcher.workspace import spend
except ImportError:  # run as a script, from its own directory
    import spend
try:
    from launcher.workspace import round_cost
except ImportError:
    try:
        import round_cost
    except ImportError:  # ag-cost's projection is not in this checkout: spend.py's prices times the minutes
        round_cost = None

HERE = Path(__file__).resolve().parent
ARMS = ("snv_indel", "with_sv")
"""workspace_pipeline.ARMS."""
ARM_DIGITS = {"snv_indel": "1", "with_sv": "2"}
"""A run's arm in its RUN_ID, as the hand-launched rounds named them (aou-wl1, aou-wl2)."""
STEPS = ("samples", "store", "phenotypes", "cohort", "measurement", "direct_sv", "fit", "score", "effects", "report", "export")
"""workspace_pipeline.STEP_NAMES (tests/test_round.py checks they agree)."""
UPSTREAM = {
    "samples": (), "store": (), "phenotypes": (), "cohort": ("samples", "phenotypes"), "measurement": ("samples", "cohort", "store"),
    "direct_sv": ("samples", "cohort", "store", "measurement"),
}
"""The steps before the fit, each with the steps whose keys its key holds (workspace_pipeline._STEPS)."""
PREPARED = tuple(UPSTREAM)
CONFIG_OVERRIDES = (
    "fit_minimum_minor_allele_frequency", "fit_minimum_r2", "fit_variant_sites", "disease_likelihood", "annotated_prior",
    "fit_region", "fit_regions", "export", "compact_fit_store", "warm_start_models",
)
"""The run config's keys a round may set apart from its base runs' without moving a step before the fit's key: the
fit's and the export's own, which no step before the fit reads (compact_fit_store keys only ag-share's fit_store step,
after direct_sv; warm_start_models, saved-model patterns such as "/mnt/disks/ops/svpgs/<earlier run>/run/fit/models/
{arm}/{trait}", starts each arm's empirical Bayes from an earlier round's model, warm_seeds: set here, it replaces the
patterns of the traits' warm_from runs and goes unchecked). A key the commit's driver does not name fails the plan."""
WARM_PATTERN = "{run}/run/fit*/models/{{arm}}/{{trait}}"
"""A warm_from run's saved models under OPS_MOUNT as warm_start_models names them (``workspace_pipeline._warm_seeds``
fills {arm} and {trait}): its completed fit's and its partial's (a FIT_ARMS run's own arm before its merge), each arm
its own."""
MEAN_FIELD = "product_mixture"
"""The draw law of a saved mean-field model, the only kind ``_warm_seeds`` starts a fit from (model.json's draw_laws)."""
ARM_VMS_CONFIG = {"cross_fit": False, "compact_fit_store": True, "arm_pass": True}
"""An arm_vms run's config apart from its traits' bases (ag-share's layout): one fold split for every trait, the fit on
the compact fit store and every trait's groups in one pass (ag-sweep's arm_pass). A key the commit's driver does not
name is left out, with a warning. cross_fit moves the cohort's key, which the store job makes anew for the union of
the traits anyway."""
SHARED_FIELDS = (
    "chromosomes", "imputed_halves", "truth_calls", "imputation_reference_ids", "crosswalk", "crosswalk_columns", "ancestry",
    "relatedness", "relatedness_columns", "fold_count", "seed", "genome_build", "strata_directory", "bubble_split",
    "genetic_map", "tandem_repeats", "gene_annotation", "codec",
)
"""The config's inputs of the samples and store steps, which a run of every trait (arm_vms) takes from its traits' base
runs only where they agree on each."""
WORKSPACE_ENVIRONMENT = (
    "PROJECT_ID", "REGION", "OPS_BUCKET", "NETWORK", "SUBNETWORK", "SERVICE_ACCOUNT", "CDR_STORAGE_PATH", "GOOGLE_PROJECT",
    "WORKSPACE_CDR", "IMPUTED_BUCKET", "PANEL_BUCKET",
)
"""submit.sh's settings that are the workspace's, the same for every job of every round: read from the environment."""
ROUNDS = "_rounds"
"""The ops bucket's directory under svpgs/ of each round's one staging (STAGED_SOURCE) and its record (round.json)."""
OPS_MOUNT = "/mnt/disks/ops/svpgs"
"""The ops bucket's svpgs/ as a job's mount shows it (run_step.sh.template)."""
COMPLETE = "_COMPLETE.json"
DIRECTORIES = ".directories.json"
"""Beside a mirrored step, its directories (run_mirror.DIRECTORIES)."""
BASES = ".bases"
"""In a run's mirror, an empty object .bases/<bucket>/<prefix> for each base mirror it reads (run_mirror.BASES)."""
LOCAL_SSD_GB = 375
UV_EXTRAS = "--extra gpu"
"""A fit job's extras, which key its shared runtime (run_step.sh.template); a CPU job has none."""
ONE_GPU = {"machine_type": "a2-highgpu-1g", "gpu_type": "nvidia-tesla-a100", "gpu_count": 1, "run_disk_type": "local-ssd", "run_disk_gb": 4 * LOCAL_SSD_GB}
"""A fit job's shape by default (ag-cost's one-GPU layout, the lead's choice for the next round): every logged arm's
resident codes, 16.6-29.6 GB [real], fit one A100-40 beside the fit's own ~4 GB, and the fused sweep is one serial
chain that a second device only shards. Four local SSDs hold an arm of the whole 547 GB store (~830-880 GB); two will
do once the compact fit store is on."""
TWO_GPU = dict(ONE_GPU, machine_type="a2-highgpu-2g", gpu_count=2)
"""Every arm's shape but the canaries' while canaries are named: the live rounds' machine, on the same disks; and
trait_jobs' default, one GPU an arm."""
ARM_VM = {"machine_type": "a2-ultragpu-1g", "gpu_type": "nvidia-a100-80gb", "gpu_count": 1, "run_disk_type": "pd-balanced", "run_disk_gb": 3500}
"""arm_vms' default (ag-share's): the arm pass holds ~47-53 GB of every trait's codes, one A100-80; every trait's LD
store and band at once, 11 x ~240-267 GB [real per trait], need ~3.5 TB of disk, pd-balanced (whose throughput grows
with its size) since how Batch mounts this machine's bundled local SSD is undocumented."""
CPU_MERGE = {"machine_type": "n2d-standard-16", "run_disk_type": "pd-balanced", "run_disk_gb": 200}
"""A merge job's shape with PARTNER_RUN (submit.sh merge, on the store job's CPU template): it restores no store, only
what finishing reads (the steps' records, the cohort and both arms' parts), so a small disk; 16 vCPUs for the report
is a choice, not a measurement."""
GPU_TYPES = {"a2-highgpu": "nvidia-tesla-a100", "a2-ultragpu": "nvidia-a100-80gb"}
"""Each A2 family's GPU, for a spec that names its machine but not the GPU."""
STORE_DISK = {"run_disk_type": "pd-balanced", "run_disk_gb": 2000}
"""arm_vms' store job's disk: the AoU rounds' run disk [real], which held the store and the steps that rewrite it
(ag-share: direct_sv's 547 GB store, the fit store's copy and measurement's outputs)."""
STORE_MACHINE = "n2d-highmem-32"
"""arm_vms' store job's machine by default (ag-share's, priced by ag-cost: the fit store's pass, one streamed CPU read
of direct_sv's store and a 20-53 GB write, ~30 min there). Its high memory is for the measurement and direct_sv steps
the job makes again for the 11-trait cohort, whose minutes on it nobody has measured."""

PHASES = (
    ("start", "queue, boot and environment", "ag-infra"),
    ("restore", "the run restored from its mirror and its base's", "ag-infra, ag-share"),
    ("store_digest", "the store's content digest", "ag-setup, ag-resume"),
    ("fit_setup", "the fit's set-up before Stage 0", "ag-setup"),
    ("stage0", "Stage 0", "ag-stage0"),
    ("band_fill", "the band, far field, moment starts and resident fill", "ag-fill, ag-stage0"),
    ("eb", "empirical Bayes", "ag-sweep, ag-eb, ag-curv, ag-prior, ag-recover"),
    ("score", "the job's scoring and effects of its own arm", "ag-score"),
)
"""A fit job's phases in order: (name, what, the agents whose patches set its minutes; ag-prof's S/eta_table.txt)."""
MERGE = ("merge", "a merge job, its start to every report", "ag-score, ag-live")
FINISH = ("finish", "the later arm's job finishing the run on its VM", "ag-score")
PREEMPTION = ("preemption", "what one preemption of the slowest job adds", "ag-restart, ag-resume")
PER_TRAIT = ("fit_setup", "stage0", "band_fill", "eb", "score")
"""The phases a run of every trait (arm_vms) goes through once a trait, in turn."""
TIMELINES = {
    # [real] Minutes (low, high) from the whole-genome rounds before and the live one (eta_budget.txt and ag-prof's
    # eta_table.txt v3, 2026-10-20): queue, boot and environment ~10; restore 17 (547 GB on pd-balanced); the store's
    # digest 12; set-up ~5; Stage 0 43-50 earlier, 65-73 live; band 9-14, moment starts 25+ and fill ~20; EB 1-5 h;
    # scoring in the merge job, 1-1.5 h with its own restore and digest; a preemption restarts the arm (~1.5 h). No job
    # of today's scores its own arm or finishes the run, so score and finish are ag-score's estimates (projected's),
    # which count only where the commit's jobs do (PARTNER_RUN).
    "today": {
        "start": (10, 10), "restore": (17, 17), "store_digest": (12, 12), "fit_setup": (5, 5), "stage0": (43, 73),
        "band_fill": (54, 59), "eb": (60, 300), "score": (4, 10), "merge": (60, 90), "finish": (3, 4), "preemption": (90, 90),
    },
    # The round's budget (eta_budget.txt; eta_table.txt v3's budget column), slowest arm: start 5, restore 8, digest 0,
    # set-up 1, Stage 0 8, band to fill 10, EB 75, score and report 15 (with PARTNER_RUN each arm scores its own
    # held-out rows in its job, 10 here, and the later one finishes the run in 5; without it a merge job of 15 does
    # both), a preemption 10.
    "budget": {
        "start": (5, 5), "restore": (8, 8), "store_digest": (0, 0), "fit_setup": (1, 1), "stage0": (8, 8),
        "band_fill": (10, 10), "eb": (75, 75), "score": (10, 10), "merge": (15, 15), "finish": (5, 5), "preemption": (10, 10),
    },
    # The owners' projections, estimates not run on a GPU or on Batch (eta_table.txt v3): start 6-9.5; restore 6.5-8.5
    # (the whole store, local SSD); digest 0 (restore receipts, not on main); set-up 0.5-5 (ag-setup: 0.5-1 once its
    # identity calls run at once, today's 5 if they do not); Stage 0 2.5-4 on local SSD; band 0.2 and fill 0-20 (20
    # without ag-fill's tee or the compact store); EB 2-28; with PARTNER_RUN, the arm's own scoring 3-8 and effects 1-2
    # on the whole store in its job, then 3-4 to push, adopt the other arm's parts, finish and push again (ag-score's);
    # without it, scoring in the merge job, 23-26 on local SSD and 35-40 on pd-balanced (ag-live's merge); a preemption
    # 16.5-24.
    "projected": {
        "start": (6, 9.5), "restore": (6.5, 8.5), "store_digest": (0, 0), "fit_setup": (0.5, 5), "stage0": (2.5, 4),
        "band_fill": (0.2, 20.2), "eb": (2, 28), "score": (4, 10), "merge": (23, 40), "finish": (3, 4), "preemption": (16.5, 24),
    },
}
"""Each phase's minutes (low, high) by timeline; a TIMELINE.json of the same form (a measured profile's) may replace one.
score is the fit job's own scoring where it scores (PARTNER_RUN, or one job of both arms); merge, a merge job's whole;
finish, the in-job merge on the later arm's VM."""
TARGET_MINUTES = 150
"""The round's wall clock goal, submission to every report (2.5 h), one preemption's minutes within it."""
SHAPE_KEYS = ("machine_type", "gpu_type", "gpu_count", "run_disk_type", "run_disk_gb")


class Layout:
    """How a round's traits and arms become runs and jobs: one run per ``runs`` ("trait_arm", "trait" or "arm"), the
    arms' runs merged per trait or for the round where ``merge``, an arm per GPU of one VM where ``parallel_arms``, and
    the fit's ``shape`` where the spec names none of its keys."""

    def __init__(self, name: str, runs: str, merge: bool, parallel_arms: bool, shape: dict, summary: str) -> None:
        self.name, self.runs, self.merge, self.parallel_arms, self.shape, self.summary = name, runs, merge, parallel_arms, shape, summary


LAYOUTS = {
    "arm_jobs": Layout("arm_jobs", "trait_arm", True, False, ONE_GPU, "one run per trait and arm (FIT_ARMS), each trait's arms merged"),
    "trait_jobs": Layout("trait_jobs", "trait", False, True, TWO_GPU, "one run per trait, both arms at once on its VM (PARALLEL_ARMS=1), no merge"),
    "arm_vms": Layout("arm_vms", "arm", True, False, ARM_VM, "a store job, then one run per arm fitting every trait (FIT_ARMS), the arms merged"),
}


class RoundError(ValueError):
    """A spec, a check or a cloud state that stops the round before anything is submitted."""


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------
def _match(pattern: str, value, what: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise RoundError(f"{what} must match {pattern}; got {value!r}")
    return value


def _shape(entry: dict, what: str) -> None:
    for key in ("machine_type", "gpu_type"):
        if key in entry:
            _match(r"[a-z0-9-]+", entry[key], f"{what}'s {key}")
    for key in ("gpu_count", "run_disk_gb"):
        if key in entry and (not isinstance(entry[key], int) or isinstance(entry[key], bool) or entry[key] < 1):
            raise RoundError(f"{what}'s {key} is a positive whole number")
    if entry.get("run_disk_type", "local-ssd") not in ("local-ssd", "pd-balanced", "pd-ssd"):
        raise RoundError(f"{what}'s run_disk_type is local-ssd, pd-balanced or pd-ssd")
    if entry.get("run_disk_type") == "local-ssd" and entry.get("run_disk_gb", LOCAL_SSD_GB) % LOCAL_SSD_GB:
        raise RoundError(f"{what}'s local-ssd run_disk_gb is a multiple of {LOCAL_SSD_GB}")


def read_spec(raw: dict) -> dict:
    """The spec with its defaults, checked: a RoundError names the first key that is wrong."""
    known = {
        "round", "source_commit", "layout", "traits", "arms", "run_prefix", "provisioning_model", "max_retries", "through",
        "fit_slot", "merge_slot", "store_slot", "hourly_usd", "env_from", "reuse", "reuse_steps", "config", "one_gpu_slowdown",
        "merge_machine_type", "merge_cpu_machine_type", "store_machine_type", "canaries", "warm_start", *SHAPE_KEYS,
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise RoundError(f"the spec has unknown keys {unknown}")
    spec = dict(raw)
    _match(r"[a-z][a-z0-9]{0,7}", spec.get("round"), "round")
    _match(r"[0-9a-f]{7,40}", spec.get("source_commit"), "source_commit")
    if spec.get("layout") not in LAYOUTS:
        raise RoundError(f"layout is one of {sorted(LAYOUTS)}")
    traits = spec.get("traits")
    if not isinstance(traits, list) or not traits:
        raise RoundError("traits lists each trait as {name, code, base_run}")
    spec.setdefault("warm_start", True)
    if not isinstance(spec["warm_start"], bool):
        raise RoundError("warm_start is true (each arm from its own earlier model) or false")
    explicit = isinstance(spec.get("config"), dict) and "warm_start_models" in spec["config"]
    for trait in traits:
        if not isinstance(trait, dict) or not {"name", "code", "base_run"} <= set(trait) or not set(trait) <= {"name", "code", "base_run", "warm_from", *SHAPE_KEYS}:
            raise RoundError(f"each trait is {{name, code, base_run}} and optionally its warm_from and its own {SHAPE_KEYS}")
        _match(r"[a-z][a-z0-9_]*", trait["name"], "a trait's name")
        _match(r"[a-z][a-z0-9]{0,2}", trait["code"], "a trait's code")
        _match(r"[a-z][a-z0-9-]{0,25}", trait["base_run"], "a trait's base_run")
        _shape(trait, trait["name"])
        if "warm_from" in trait and (not spec["warm_start"] or explicit):
            raise RoundError(f"{trait['name']}'s warm_from names runs where " + ("config sets warm_start_models" if explicit else "warm_start is false"))
        if not isinstance(trait.get("warm_from", []), list) or len(set(trait.get("warm_from", []))) != len(trait.get("warm_from", [])):
            raise RoundError(f"{trait['name']}'s warm_from lists runs, each once")
        for run in trait.get("warm_from", []):
            _match(r"[a-z][a-z0-9-]{0,25}", run, f"a run of {trait['name']}'s warm_from")
    spec["traits"] = traits = [dict(trait, warm_from=list(trait.get("warm_from", [trait["base_run"]] if spec["warm_start"] and not explicit else []))) for trait in traits]
    for key in ("name", "code"):
        values = [trait[key] for trait in traits]
        if len(set(values)) != len(values):
            raise RoundError(f"each trait's {key} is its own")
    spec.setdefault("arms", list(ARMS))
    if not spec["arms"] or not set(spec["arms"]) <= set(ARMS) or len(set(spec["arms"])) != len(spec["arms"]):
        raise RoundError(f"arms are some of {ARMS}")
    spec["arms"] = [arm for arm in ARMS if arm in spec["arms"]]
    spec.setdefault("run_prefix", f"aou-{spec['round']}")
    _match(r"[a-z][a-z0-9-]{0,19}", spec["run_prefix"], "run_prefix")
    default = LAYOUTS[spec["layout"]].shape
    spec.setdefault("machine_type", default["machine_type"])
    if isinstance(spec["machine_type"], str):
        spec.setdefault("gpu_count", _gpus(spec["machine_type"], default["gpu_count"]))
        spec.setdefault("gpu_type", GPU_TYPES.get(spec["machine_type"].rsplit("-", 1)[0], default["gpu_type"]))
    for key in SHAPE_KEYS:
        spec.setdefault(key, default[key])
    _shape(spec, "the round")
    spec.setdefault("canaries", [])
    if not isinstance(spec["canaries"], list):
        raise RoundError("canaries lists arms as TRAIT:ARM")
    by_name = {**{trait["name"]: trait for trait in traits}, **{trait["code"]: trait for trait in traits}}
    canaries = []
    for entry in spec["canaries"]:
        trait, _colon, arm = entry.partition(":") if isinstance(entry, str) else ("", "", "")
        if trait not in by_name or arm not in spec["arms"]:
            raise RoundError(f"a canary is TRAIT:ARM, a trait's code or name and one of the round's arms; got {entry!r}")
        canaries.append(f"{by_name[trait]['code']}:{arm}")
    if len(set(canaries)) != len(canaries):
        raise RoundError("each canary is named once")
    if canaries and LAYOUTS[spec["layout"]].runs != "trait_arm":
        raise RoundError("canaries are arms of arm_jobs, whose runs fit one arm each")
    spec["canaries"] = canaries
    spec.setdefault("provisioning_model", "SPOT")
    if spec["provisioning_model"] not in ("SPOT", "STANDARD"):
        raise RoundError("provisioning_model is SPOT or STANDARD")
    spec.setdefault("max_retries", 3 if spec["provisioning_model"] == "SPOT" else 0)
    if not isinstance(spec["max_retries"], int) or not 0 <= spec["max_retries"] <= 10:
        raise RoundError("max_retries is 0 to 10 (Batch's limit)")
    spec.setdefault("through", "report")
    if spec["through"] not in STEPS[STEPS.index("fit"):]:
        raise RoundError("through is the fit or a later step")
    spec.setdefault("fit_slot", "train-1")
    spec.setdefault("merge_slot", "train-2")
    spec.setdefault("store_slot", "prepare")
    for key in ("fit_slot", "merge_slot", "store_slot"):
        _match(r"[a-z][a-z0-9-]{0,20}", spec[key], key)
    if spec["fit_slot"] == spec["merge_slot"]:
        raise RoundError("a merge job goes on its own slot: the fit's slot still names each run's fit job")
    spec.setdefault("env_from", traits[0]["base_run"])
    _match(r"[a-z][a-z0-9-]{0,25}", spec["env_from"], "env_from")
    spec.setdefault("reuse", "base")
    if spec["reuse"] not in ("base", "copy"):
        raise RoundError("reuse is base (BASE_RUN) or copy")
    spec.setdefault("reuse_steps", list(PREPARED))
    if not isinstance(spec["reuse_steps"], list) or not set(spec["reuse_steps"]) <= set(PREPARED):
        raise RoundError(f"reuse_steps names some of {PREPARED}")
    spec.setdefault("config", {})
    if not isinstance(spec["config"], dict) or not set(spec["config"]) <= set(CONFIG_OVERRIDES):
        raise RoundError(f"config sets only {CONFIG_OVERRIDES}: any other key moves a reused step's key")
    spec.setdefault("one_gpu_slowdown", 1.0)
    if not isinstance(spec["one_gpu_slowdown"], (int, float)) or spec["one_gpu_slowdown"] < 1:
        raise RoundError("one_gpu_slowdown is at least 1")
    spec.setdefault("merge_machine_type", None)
    spec.setdefault("merge_cpu_machine_type", CPU_MERGE["machine_type"])
    for key in ("merge_machine_type", "merge_cpu_machine_type"):
        if spec[key] is not None:
            _match(r"[a-z0-9-]+", spec[key], key)
    spec.setdefault("store_machine_type", STORE_MACHINE)
    _match(r"[a-z0-9-]+", spec["store_machine_type"], "store_machine_type (arm_vms' store job's CPU machine)")
    if spec.get("hourly_usd") is not None and (not isinstance(spec["hourly_usd"], int) or spec["hourly_usd"] < 1):
        raise RoundError("hourly_usd is a whole number of dollars (budget_guard's rate)")
    return spec


def runs_of(spec: dict) -> list:
    """The round's runs, in the layout's order: (RUN_ID, its traits, its arms)."""
    prefix, traits, arms = spec["run_prefix"], spec["traits"], spec["arms"]
    runs = LAYOUTS[spec["layout"]].runs
    if runs == "trait_arm":
        listed = [(f"{prefix}-{trait['code']}{ARM_DIGITS[arm]}", [trait], [arm]) for trait in traits for arm in arms]
    elif runs == "trait":
        listed = [(f"{prefix}-{trait['code']}", [trait], list(arms)) for trait in traits]
    else:
        listed = [(f"{prefix}-all{ARM_DIGITS[arm]}", list(traits), [arm]) for arm in arms]
    for run_id, _traits, _arms in listed:
        _match(r"[a-z][a-z0-9-]{0,25}", run_id, "a RUN_ID (run_prefix and the trait codes)")
    return listed


def shape_of(spec: dict, traits: list, arms: list) -> dict:
    """A run's machine and disk: the round's, but TWO_GPU's for a run of an arm that is not a canary while canaries are
    named, and a run of one trait with keys of its own takes those over either."""
    shape = {key: spec[key] for key in SHAPE_KEYS}
    if spec["canaries"] and not (len(traits) == 1 and all(f"{traits[0]['code']}:{arm}" in spec["canaries"] for arm in arms)):
        shape = dict(TWO_GPU)
    if len(traits) == 1:
        shape.update({key: traits[0][key] for key in SHAPE_KEYS if key in traits[0]})
    return shape


# ---------------------------------------------------------------------------
# Timelines and prices
# ---------------------------------------------------------------------------
def read_timeline(name: str) -> tuple:
    """A timeline by name or from a JSON file of {phase: [low, high]}: (its label, its minutes)."""
    if name in TIMELINES:
        return name, TIMELINES[name]
    raw = json.loads(Path(name).read_text(encoding="utf-8"))
    phases = [phase for phase, _what, _owner in PHASES] + [MERGE[0], FINISH[0], PREEMPTION[0]]
    missing = sorted(set(phases) - set(raw))
    if missing:
        raise RoundError(f"the timeline {name} lacks {missing}")
    return Path(name).name, {phase: (float(raw[phase][0]), float(raw[phase][1])) for phase in phases}


def hourly(machine: str, model: str, disk_type: str, disk_gb: int) -> float:
    rate, unpriced = spend.usd_per_hour(machine, model, [{"type": disk_type, "sizeGb": disk_gb}])
    if rate is None or unpriced:
        raise RoundError(f"no price for {machine} {model} with {disk_gb} GB {disk_type}: {unpriced} (launcher/workspace/spend.py)")
    return rate


def fit_rows(timeline: dict, spec: dict, arms_in_turn: bool = False, in_job: bool = True) -> list:
    """A fit job's phases on ``timeline`` for the spec's layout: (name, low, high, note). trait_jobs' EB is one GPU's
    (one_gpu_slowdown), or both arms' in turn where the commit has no PARALLEL_ARMS (``arms_in_turn``), and its job
    scores and finishes the run itself; an arm's job scores its own arm only where the later one finishes the run
    (``in_job``: the commit's PARTNER_RUN), its merge job scoring both otherwise; arm_vms' run goes through each trait's
    set-up, Stage 0, band, EB and scoring in turn."""
    layout = LAYOUTS[spec["layout"]]
    rows = []
    for phase, _what, _owner in PHASES:
        (low, high), note = timeline[phase], ""
        if layout.parallel_arms and arms_in_turn and phase in PER_TRAIT:
            low, high, note = low * len(spec["arms"]), high * len(spec["arms"]), f"x{len(spec['arms'])} arms in turn: no PARALLEL_ARMS in the commit"
        elif phase == "eb" and layout.parallel_arms and spec["one_gpu_slowdown"] != 1:
            low, high, note = low * spec["one_gpu_slowdown"], high * spec["one_gpu_slowdown"], f"x{spec['one_gpu_slowdown']:g}: an arm on one GPU"
        if phase == "score" and not layout.merge:
            low, high, note = low + timeline[FINISH[0]][0], high + timeline[FINISH[0]][1], "and the run's finish, in the job"
        elif phase == "score" and not in_job:
            low, high, note = 0, 0, "in the merge job: the commit has no PARTNER_RUN"
        if layout.runs == "arm" and phase in PER_TRAIT:
            count = len(spec["traits"])
            low, high, note = low * count, high * count, f"x{count} traits in turn"
        rows.append((phase, low, high, note))
    return rows


# ---------------------------------------------------------------------------
# The cloud
# ---------------------------------------------------------------------------
class Gcloud:
    """The workspace's gcloud, round.py's one way to the cloud. Reads in any mode; a change (a copy, a write, a
    submission) only with --execute."""

    def __init__(self, bucket: str, execute: bool) -> None:
        self.bucket, self.execute = bucket, execute

    def url(self, path: str) -> str:
        return f"gs://{self.bucket}/svpgs/{path}"

    def _run(self, arguments: list, change: bool = False) -> subprocess.CompletedProcess:
        if change and not self.execute:
            raise RuntimeError("a change to the cloud outside --execute")
        done = subprocess.run(["gcloud", *arguments], capture_output=True, text=True, check=False)
        if change and done.returncode != 0:
            raise RoundError(f"gcloud {' '.join(arguments[:2])} failed: {done.stderr.strip()[-500:]}")
        return done

    def read(self, path: str):
        """An object's text, or None where there is none."""
        shown = self._run(["storage", "cat", self.url(path)])
        return shown.stdout if shown.returncode == 0 else None

    def listing(self, path: str) -> dict:
        """Every object under the directory ``path``: its name relative to it, and its size."""
        root = self.url(path.rstrip("/") + "/")
        shown = self._run(["storage", "ls", "-l", "-r", root])
        found = {}
        for line in shown.stdout.splitlines() if shown.returncode == 0 else ():
            matched = re.match(r"^\s*(\d+)\s+\S+\s+(gs://\S+)$", line)
            if matched and matched.group(2).startswith(root) and not matched.group(2).endswith("/"):
                found[matched.group(2)[len(root):]] = int(matched.group(1))
        return found

    def size(self, path: str):
        """One object's size, or None where there is none."""
        shown = self._run(["storage", "ls", "-l", self.url(path)])
        matched = re.match(r"^\s*(\d+)\s", shown.stdout) if shown.returncode == 0 else None
        return int(matched.group(1)) if matched else None

    def children(self, path: str) -> set:
        """The names directly under the directory ``path`` (one listing, not recursive)."""
        root = self.url(path.rstrip("/") + "/") if path else self.url("")
        shown = self._run(["storage", "ls", root])
        lines = shown.stdout.splitlines() if shown.returncode == 0 else []
        return {line[len(root):].strip("/") for line in lines if line.startswith(root) and line[len(root):].strip("/")}

    def completed(self, run: str) -> set:
        """The steps a run's mirror holds completed: one listing of its run/*/_COMPLETE.json."""
        shown = self._run(["storage", "ls", self.url(f"{run}/run/*/{COMPLETE}")])
        lines = shown.stdout.splitlines() if shown.returncode == 0 else []
        return {matched.group(1) for line in lines for matched in [re.search(r"/run/([^/]+)/" + re.escape(COMPLETE) + "$", line.strip())] if matched}

    def models(self, run: str) -> dict:
        """The saved models a run's mirror holds where warm_start_models looks (WARM_PATTERN): each model's directory
        under svpgs/ by (arm, trait), from one listing of their model.json."""
        shown = self._run(["storage", "ls", self.url(WARM_PATTERN.format(run=run).format(arm="*", trait="*") + "/model.json")])
        found: dict = {}
        for line in shown.stdout.splitlines() if shown.returncode == 0 else ():
            matched = re.search(r"/svpgs/(" + re.escape(run) + r"/run/fit[^/]*/models/([^/]+)/([^/]+))/model\.json$", line.strip())
            if matched:
                found.setdefault((matched.group(2), matched.group(3)), []).append(matched.group(1))
        return found

    def download(self, path: str, local: Path):
        """A directory (code: a staged source tree) copied into ``local``, a read: the local tree where it came whole,
        object by object at its size, else None."""
        local.mkdir(parents=True)
        done = self._run(["storage", "cp", "-r", self.url(path), f"{local}/"])
        tree, held = local / Path(path).name, self.listing(path)
        whole = done.returncode == 0 and bool(held) and all((tree / name).is_file() and (tree / name).stat().st_size == size for name, size in held.items())
        return tree if whole else None

    def copy(self, source: str, target: str) -> None:
        """One object, server-side."""
        self._run(["storage", "cp", self.url(source), self.url(target)], change=True)

    def copy_tree(self, source: str, target: str, skip=()) -> None:
        """A directory, server-side, entry by entry at its top (a directory with cp -r, an object with cp), but the
        top-level objects named in ``skip``, which the caller copies after checking the rest."""
        entries: dict = {}
        for name in self.listing(source):
            first, _slash, rest = name.partition("/")
            entries[first] = entries.get(first, False) or bool(rest)
        for name, directory in sorted(entries.items()):
            if directory:
                self._run(["storage", "cp", "-r", self.url(f"{source}/{name}"), self.url(target) + "/"], change=True)
            elif name not in skip:
                self.copy(f"{source}/{name}", f"{target}/{name}")

    def upload(self, local: Path, parent: str) -> None:
        """A local directory into the directory ``parent``, under its own name."""
        self._run(["storage", "cp", "-r", str(local), self.url(parent) + "/"], change=True)

    def write(self, target: str, text: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "object"
            path.write_text(text, encoding="utf-8")
            self._run(["storage", "cp", str(path), self.url(target)], change=True)

    def job_state(self, name: str, project: str, region: str):
        shown = self._run(["batch", "jobs", "describe", name, f"--project={project}", f"--location={region}", "--format=value(status.state)"])
        return (shown.stdout.strip() or None) if shown.returncode == 0 else None


# ---------------------------------------------------------------------------
# The source, the base runs and their steps
# ---------------------------------------------------------------------------
def git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(["git", "-C", str(repository), *arguments], capture_output=True, check=True).stdout


def tree_files(root: Path) -> dict:
    """Every file under ``root``: its size and SHA-256 digest by relative name."""
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*")) if path.is_file() and not path.is_symlink()
    }


def commit_tree(repository: Path, commit: str, into: Path) -> dict:
    """The commit's tree (git archive) extracted into ``into``, and its files (``tree_files``)."""
    archive = git(repository, "archive", "--format=tar", commit)
    into.mkdir(parents=True)
    subprocess.run(["tar", "-x", "-C", str(into)], input=archive, check=True)
    return tree_files(into)


def manifest(files: dict) -> str:
    """source.sha256: sha256sum's check form, one line a file (run_step.sh.template checks a job's copy with it)."""
    return "".join(f"{digest}  {name}\n" for name, (_size, digest) in sorted(files.items()))


def runtime_key(tree: Path, extras: str = UV_EXTRAS) -> str:
    """The shared runtime's key for a staged tree (run_step.sh.template: uv.lock, pyproject.toml and the extras)."""
    content = (tree / "uv.lock").read_bytes() + (tree / "pyproject.toml").read_bytes() + (extras + "\n").encode()
    return hashlib.sha256(content).hexdigest()[:16]


def holders(cloud: Gcloud, base: str, steps: list) -> tuple:
    """For each of ``steps``, the run whose mirror holds it completed: ``base``'s own, or a base it names (BASES), and
    those the bases name in turn, as a job's restore reads them (run_mirror.bases_of). And the bases named in another
    bucket, which this does not follow."""
    held, elsewhere, seen, pending = {}, [], set(), [base]
    while pending and len(held) < len(steps):
        run = pending.pop(0)
        if run in seen:
            continue
        seen.add(run)
        complete = cloud.completed(run)
        for step in steps:
            if step not in held and step in complete:
                held[step] = run
        for name in sorted(cloud.listing(f"{run}/run/{BASES}")):
            named = re.fullmatch(r"([^/]+)/svpgs/([a-z][a-z0-9-]{0,25})/run", name)
            if named and named.group(1) == cloud.bucket:
                pending.append(named.group(2))
            else:
                elsewhere.append(name)
    return held, elsewhere


def source_of(cloud: Gcloud, run: str):
    """The directory under svpgs/ of the source a run's jobs ran: its own staging, or the round's its STAGED_SOURCE
    object names (written by ``execute``); None where neither is there."""
    if cloud.read(f"{run}/source/pyproject.toml") is not None:
        return f"{run}/source"
    named = cloud.read(f"{run}/STAGED_SOURCE")
    return f"{named.strip()}/source" if named else None


def same_code(commit: Path, base: Path, lists=None) -> dict:
    """Whether each step before the fit has the same code in the two trees: every step where their sv_pgs are the same
    file for file; otherwise as the commit's launcher/workspace/step_codes.py --before-fit says (it reads the trees'
    sources, imports nothing), given the run's lists of diseases and traits (``lists``) where they are known, so the
    phenotype queries a run never runs do not count. {} where neither can tell."""
    if {name: entry for name, entry in tree_files(commit).items() if name.startswith("sv_pgs/")} == {
        name: entry for name, entry in tree_files(base).items() if name.startswith("sv_pgs/")
    }:
        return {step: True for step in PREPARED}
    tool = commit / "launcher/workspace/step_codes.py"
    if not tool.is_file():
        return {}
    with tempfile.TemporaryDirectory() as temporary:
        named = []
        if lists is not None:
            (Path(temporary) / "run_config.json").write_text(json.dumps(lists), encoding="utf-8")
            named = ["--config", str(Path(temporary) / "run_config.json")]
        shown = subprocess.run([sys.executable, str(tool), "--before-fit", *named, str(base), str(commit)], capture_output=True, text=True, check=False)
    found = {}
    for line in shown.stdout.splitlines():
        matched = re.match(r"^(\w+)\s+(same|differs)\b", line)
        if matched and matched.group(1) in PREPARED:
            found[matched.group(1)] = matched.group(2) == "same"
    return found if set(found) == set(PREPARED) else {}


def reusable(wanted: list, complete: set, code: dict) -> tuple:
    """The steps of ``wanted`` a run takes from its base, and why each other one is its job's to make: a step's key
    holds its code and its upstream steps' keys, so it is taken only where it is complete in the base, its code is the
    same (``code``: step -> True or False; absent, unknown) and every step it reads is taken too."""
    taken, made = [], {}
    for step in PREPARED:
        blocked = [up for up in UPSTREAM[step] if up not in taken]
        if step not in wanted:
            made[step] = "not asked"
        elif step not in complete:
            made[step] = "not complete in the base"
        elif code.get(step) is None:
            made[step] = "its code unchecked"
        elif not code[step]:
            made[step] = "its code differs from the base's"
        elif blocked:
            made[step] = f"reads {', '.join(blocked)}, made again"
        else:
            taken.append(step)
    return taken, made


def record_key(cloud: Gcloud, run, step: str = "store"):
    """The key in ``run``'s mirror's record of ``step`` (its _COMPLETE.json), or None where there is none."""
    text = cloud.read(f"{run}/run/{step}/{COMPLETE}") if run else None
    try:
        return json.loads(text)["key"] if text else None
    except (ValueError, KeyError, TypeError):
        return None


def warm_start(cloud: Gcloud, trait: dict, arm: str, store, saved: dict, keys: dict, metadata: dict, model_format=None) -> tuple:
    """Where ``arm``'s fit of ``trait`` starts: ("warm", the first of its warm_from runs' saved models that
    ``workspace_pipeline._warm_seeds`` can take, and how many there are) or ("cold", why each run gives none). A model
    qualifies in a run whose store record has the key of the store this run keeps (``store``: the same store has the
    same sites, which _warm_seeds requires) where its model.json lists a mean-field draw law and is of the commit's
    ``model_format`` (artifact.MODEL_FORMAT, the only one load_model reads; unchecked where None). The job ranks every model
    that qualifies (``_seed_rank``: the same arm's reliabilities, the same problem, an accepted fit, the newer), so with
    several it may take another than the one named. ``saved``, ``keys`` and ``metadata`` keep the bucket's reads for the
    other arms and runs."""
    found, reasons = [], []
    for run in trait["warm_from"]:
        if run not in saved:
            saved[run] = cloud.models(run)
        folders = saved[run].get((arm, trait["name"]), [])
        if not folders:
            reasons.append(f"no saved model in {run}")
            continue
        if run not in keys:
            keys[run] = record_key(cloud, holders(cloud, run, ["store"])[0].get("store"))
        if store is None or keys[run] != store:
            reasons.append(f"{run}'s store is not the one this run keeps")
            continue
        for folder in folders:
            if folder not in metadata:
                text = cloud.read(f"{folder}/model.json")
                try:
                    metadata[folder] = json.loads(text) if text else None
                except ValueError:
                    metadata[folder] = None
            shown = metadata[folder]
            if not isinstance(shown, dict):
                reasons.append(f"{folder}/model.json unreadable")
            elif model_format is not None and shown.get("format") != model_format:
                reasons.append(f"{folder} is a {shown.get('format')!r} model, not {model_format!r}")
            elif MEAN_FIELD not in shown.get("draw_laws", ()):
                reasons.append(f"{folder} holds no mean-field model")
            else:
                found.append((folder, bool((shown.get("acceptance") or {}).get("accepted", False))))
    if not found:
        return "cold", "; ".join(reasons) or "no warm_from run named"
    folder, accepted = found[0]
    return "warm", f"svpgs/{folder} ({'an accepted fit' if accepted else 'a fit not accepted'}" + (f"; {len(found)} models, the job takes the best)" if len(found) > 1 else ")")


def warm_patterns(traits: list) -> list:
    """warm_start_models for a run of ``traits``: WARM_PATTERN of each trait's warm_from runs, in order, each once."""
    patterns: list = []
    for trait in traits:
        for run in trait["warm_from"]:
            pattern = f"{OPS_MOUNT}/" + WARM_PATTERN.format(run=run)
            if pattern not in patterns:
                patterns.append(pattern)
    return patterns


def run_config(bases: dict, run_id: str, traits: list, spec: dict, accept_from, warm: bool = False) -> dict:
    """A run's config: its base run's (``bases``: base run -> its config), with its own marker directory, the round's
    fit-only keys, accept_steps_from where the commit's driver reads it, warm_start_models of its traits' warm_from runs
    where ``warm`` (the commit's driver reads it and the spec's config names none), and for a run of several traits each
    trait from its own base's lists (``SHARED_FIELDS`` checked equal by the caller)."""
    config = json.loads(json.dumps(bases[traits[0]["base_run"]]))
    config["marker_directory"] = f"{OPS_MOUNT}/{run_id}/markers"
    if len(traits) > 1:
        names = {trait["name"] for trait in traits}
        for key in ("diseases", "traits"):
            listed = []
            for trait in traits:
                listed += [name for name in bases[trait["base_run"]].get(key, []) if name in names and name not in listed]
            config[key] = listed
        weights = {}
        for trait in traits:
            weights.update({name: weight for name, weight in ((bases[trait["base_run"]].get("panel") or {}).get("weights") or {}).items() if name in names})
        if config.get("panel"):
            config["panel"] = dict(config["panel"], weights=weights)
    # A base a round made carries that round's accept_steps_from and warm starts, neither of them this run's.
    config.pop("accept_steps_from", None)
    config.pop("warm_start_models", None)
    config.update(spec["config"])
    if accept_from is not None:
        config["accept_steps_from"] = accept_from
    if warm and warm_patterns(traits):
        config["warm_start_models"] = warm_patterns(traits)
    return config


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
class Plan:
    """The round as planned: its runs (each with the steps it keeps and remakes and, where ``warm``, each arm's start)
    and fit jobs; arm_vms' store job (``store``); with PARTNER_RUN each trait's in-job finish and its CPU merge job for
    --follow (``finishes``), without it each trait's merge job (``merges``); the copies, the checks and the notes."""

    def __init__(self, spec: dict, timeline_label: str, timeline: dict) -> None:
        self.spec, self.timeline_label, self.timeline = spec, timeline_label, timeline
        self.staging = f"{ROUNDS}/{spec['round']}"
        self.work = Path(tempfile.mkdtemp(prefix="round-"))
        self.files: dict = {}
        self.runs: list = []
        self.jobs: list = []
        self.store = None
        self.merges: list = []
        self.finishes: list = []
        self.copies: list = []
        self.checks: list = []
        self.notes: list = []
        self.arms_in_turn = False
        self.partner = False
        self.warm = False

    def check(self, passed, text: str, failure: str = "FAIL") -> None:
        self.checks.append(("ok" if passed else failure, text))


def job_environment(spec: dict, run_id: str, slot: str, shape: dict, staging: str, base, through=None, gpu: bool = True, **own) -> dict:
    """submit.sh's variables for one job, but the workspace's (WORKSPACE_ENVIRONMENT): ``shape``'s machine and disk, its
    GPUs and the fit's extras where ``gpu`` (a CPU job, submit.sh store or merge, has neither), and the job's ``own``
    (FIT_ARMS, PARALLEL_ARMS, PARTNER_RUN) where set."""
    rate = hourly(shape["machine_type"], spec["provisioning_model"], shape["run_disk_type"], shape["run_disk_gb"])
    environment = {"RUN_ID": run_id, "BUDGET_SLOT": slot, "THROUGH": through or spec["through"], "STAGED_SOURCE": staging, "MACHINE_TYPE": shape["machine_type"]}
    if gpu:
        environment.update(GPU_TYPE=shape["gpu_type"], GPU_COUNT=str(shape["gpu_count"]))
    environment.update({
        "PROVISIONING_MODEL": spec["provisioning_model"], "MAX_RETRIES": str(spec["max_retries"]), "RUN_DISK_TYPE": shape["run_disk_type"],
        "RUN_DISK_GB": str(shape["run_disk_gb"]), "SHARED_RUNTIME": "1", "UV_EXTRAS": UV_EXTRAS if gpu else "",
        "HOURLY_USD": str((spec.get("hourly_usd") if gpu else None) or math.ceil(rate)),
    })
    if base is not None:
        environment["BASE_RUN"] = base
    environment.update({name: value for name, value in own.items() if value})
    return environment


def _job(run_id: str, slot: str, command: str, environment: dict, after=()) -> dict:
    """One job: submit.sh's ``command`` (fit, store or merge) with ``environment``, its rate, and the jobs it waits for."""
    rate = hourly(environment["MACHINE_TYPE"], environment["PROVISIONING_MODEL"], environment["RUN_DISK_TYPE"], int(environment["RUN_DISK_GB"]))
    return {"name": f"svpgs-{run_id}-{slot}", "run_id": run_id, "command": command, "environment": environment, "rate": rate, "after": list(after)}


def _gpus(machine: str, default: int) -> int:
    """An A2 machine's GPUs from its name (a2-highgpu-2g: 2), else ``default``."""
    matched = re.fullmatch(r"a2-\w+-(\d+)g", machine)
    return int(matched.group(1)) if matched else default


def build(spec: dict, timeline_label: str, timeline: dict, cloud, repository: Path) -> Plan:
    """The round's plan, with every check that reads (the bucket's, unless ``cloud`` is None: --offline)."""
    plan = Plan(spec, timeline_label, timeline)
    layout = LAYOUTS[spec["layout"]]
    commit = plan.work / "source"
    try:
        plan.files = commit_tree(repository, spec["source_commit"], commit)
    except (subprocess.CalledProcessError, FileNotFoundError):
        plan.check(False, f"source: commit {spec['source_commit']} is not in {repository}: fetch it first")
    driver = ""
    if plan.files:
        size = sum(size for size, _digest in plan.files.values())
        plan.notes.append(f"source: {spec['source_commit']}, {len(plan.files)} files, {size / 1e6:.1f} MB, staged once at svpgs/{plan.staging}/")
        launcher = sorted(name for name in plan.files if name.startswith("launcher/workspace/") and name != "launcher/workspace/round.py")
        differs = [name for name in launcher if not (repository / name).is_file() or hashlib.sha256((repository / name).read_bytes()).hexdigest() != plan.files[name][1]]
        plan.check(not differs, "launcher: this checkout's launcher/workspace " + (f"differs from the commit's: {', '.join(differs)}" if differs else "is the commit's (submit.sh renders each job's task from it)"))
        task = (commit / "launcher/workspace/run_step.sh.template").read_text(encoding="utf-8")
        driver = (commit / "sv_pgs/workspace_pipeline.py").read_text(encoding="utf-8")
        plan.check("STAGED_SOURCE" in task, "launcher: the commit's task reads STAGED_SOURCE")
        if layout.parallel_arms:
            plan.arms_in_turn = "PARALLEL_ARMS" not in (commit / "launcher/workspace/submit.sh").read_text(encoding="utf-8")
            shown = "lacks PARALLEL_ARMS, so each job fits its arms in turn" if plan.arms_in_turn else "has PARALLEL_ARMS"
            plan.check(not plan.arms_in_turn, f"layout: the commit's submit.sh {shown}", failure="warn")
        if spec["reuse"] == "base" or layout.runs == "arm":
            plan.check("--base" in task, "reuse: the commit's task restores from BASE_RUN" + ("" if layout.runs == "arm" else " (else reuse copy)"))
        if layout.merge and len(spec["arms"]) > 1:
            finishing = spec["through"] in ("report", "export")
            plan.partner = finishing and "PARTNER_RUN" in task
            shown = ("the later arm's job finishes the run on its VM (PARTNER_RUN); a CPU merge job only where neither did (--follow)" if plan.partner
                     else "a merge job after both arms (--follow): " + ("the commit's task has no PARTNER_RUN" if finishing else f"through {spec['through']} stops before the report"))
            plan.check(plan.partner, f"merge: {shown}", failure="warn")
        if spec["config"]:
            unknown = sorted(key for key in spec["config"] if not re.search(rf"\b{key}\b", driver))
            plan.check(not unknown, "config: the commit's driver " + (f"does not name {', '.join(unknown)}" if unknown else "names every key the round sets"))
        if any(trait["warm_from"] for trait in spec["traits"]):
            plan.check("warm_start_models" in driver, "warm starts: the commit's driver " + (
                "has warm_start_models: each arm from its own arm's saved model in its trait's warm_from runs" if "warm_start_models" in driver
                else "has no warm_start_models: every arm starts cold"), failure="warn")
    if "warm_start_models" in spec["config"]:
        plan.notes.append("warm starts: the spec's config names warm_start_models for every run; its models are not checked")
    elif not spec["warm_start"]:
        plan.notes.append("warm starts: off (warm_start false), every arm starts cold")
    warm = plan.warm = bool(plan.files) and "warm_start_models" in driver and any(trait["warm_from"] for trait in spec["traits"])
    saved: dict = {}
    keys: dict = {}
    metadata: dict = {}
    records: dict = {}
    artifact = commit / "sv_pgs/artifact.py"
    model_format = re.search(r'^MODEL_FORMAT = "([^"]+)"', artifact.read_text(encoding="utf-8"), re.M) if artifact.is_file() else None
    model_format = model_format.group(1) if model_format else None

    def kept_store(holder):
        """The key of the store record ``holder``'s mirror holds (the store a run keeps), read once a holder."""
        if holder not in records:
            records[holder] = record_key(cloud, holder)
        return records[holder]

    shapes = [shape_of(spec, [trait], [arm]) for trait in spec["traits"] for arm in spec["arms"]] if layout.runs == "trait_arm" else []
    if spec["canaries"]:
        plan.notes.append(f"canaries: {len(spec['canaries'])} of {len(shapes)} arms on {spec['machine_type']} ({', '.join(spec['canaries'])}), the others on "
                          f"{TWO_GPU['machine_type']}: put every arm on {spec['machine_type']} once the canaries' Stage 0, fill and host memory lines are read")
    elif layout.runs == "trait_arm" and spec["gpu_count"] == 1:
        plan.check(False, f"shape: every arm on one GPU ({spec['machine_type']}), where no arm has run yet: name two canaries first (\"canaries\")", failure="warn")
    accepts = bool(plan.files) and "accept_steps_from" in driver
    arm_keys: dict = {}
    if layout.runs == "arm" and plan.files:
        arm_keys = {key: value for key, value in ARM_VMS_CONFIG.items() if re.search(rf"\b{key}\b", driver)}
        missing = [key for key in ARM_VMS_CONFIG if key not in arm_keys]
        plan.check(not missing, "arm_vms: the commit's driver has " + (f"no {', '.join(missing)}, left out of the runs' config" if missing else ", ".join(ARM_VMS_CONFIG)), failure="warn")
    listed = runs_of(spec)
    store_id = None
    if layout.runs == "arm":
        store_id = f"{spec['run_prefix']}-all0"
        listed = [(store_id, list(spec["traits"]), [])] + listed
    bases: dict = {}
    if cloud is None:
        plan.check(False, "offline: the bucket was not read (the base runs, env.tar, the shared runtime, the staging, the runs)", failure="warn")
    else:
        if plan.files:
            key = runtime_key(commit)
            published = any(name.endswith("COMPLETE") for name in cloud.listing(f"_runtime/{key}"))
            plan.check(published, f"runtime: a shared runtime is published for key {key}" + ("" if published else ": none; the first job of the round builds and publishes it"), failure="warn")
            same = all(cloud.read(f"{spec['env_from']}/source/{name}") == (commit / name).read_text(encoding="utf-8") for name in ("uv.lock", "pyproject.toml"))
            plan.check(same, f"env.tar: {spec['env_from']}'s staged uv.lock and pyproject.toml are the commit's" + ("" if same else ": stage.sh a run at this commit and name it env_from"))
            staged = cloud.listing(f"{plan.staging}/source")
            recorded = cloud.read(f"{plan.staging}/source.sha256")
            expected = {name: size for name, (size, _digest) in plan.files.items()}
            fresh, ours = not staged and recorded is None, recorded == manifest(plan.files)
            plan.check(fresh or ours, f"staging: svpgs/{plan.staging}/source " + (
                "is new" if fresh else ("holds this tree; kept" if staged == expected else "holds part of this tree; completed") if ours else "holds another tree: name another round"))
        present = cloud.children("")
        existing = [run_id for run_id, _traits, _arms in listed if run_id in present]
        plan.check(not existing, f"runs: the round's {len(listed)} runs are new" + (f"; these exist already: {', '.join(existing)}" if existing else ""))
        for base in sorted({trait["base_run"] for trait in spec["traits"]}):
            text = cloud.read(f"{base}/run_config.json")
            if text is not None:
                bases[base] = json.loads(text)
        missing = sorted({trait["base_run"] for trait in spec["traits"]} - set(bases))
        plan.check(not missing, "bases: each has its run_config.json" + (f"; these have none: {', '.join(missing)}" if missing else ""))
    trees: dict = {}
    downloaded: dict = {}
    found: dict = {}
    for run_id, traits, arms in listed:
        shape = shape_of(spec, traits, arms)
        if store_id is not None and run_id != store_id:
            # An arm's run of arm_vms: every step before the fit is the store job's, restored through BASE_RUN once the
            # job has SUCCEEDED (--follow), in the store run's config.
            store = plan.runs[0]
            config = dict(store["config"], marker_directory=f"{OPS_MOUNT}/{run_id}/markers") if store["config"] else None
            starts = None
            if warm and cloud is not None:
                kept = kept_store(store["holders"].get("store"))
                starts = {(trait["name"], arm): warm_start(cloud, trait, arm, kept, saved, keys, metadata, model_format) for trait in spec["traits"] for arm in arms}
            plan.runs.append({"run_id": run_id, "traits": store["traits"], "arms": arms, "base": store_id, "reuse": list(PREPARED), "made": {},
                              "config": config, "shape": shape, "holders": {}, "starts": starts})
            environment = job_environment(spec, run_id, spec["fit_slot"], shape, plan.staging, store_id, FIT_ARMS=" ".join(arms))
            plan.jobs.append(_job(run_id, spec["fit_slot"], "fit", environment, after=[plan.store["name"]]))
            continue
        base = traits[0]["base_run"]
        wanted = list(spec["reuse_steps"])
        if len(traits) > 1:
            # A run of several traits has its own phenotypes and every step after them: no trait's base holds them.
            wanted = [step for step in wanted if step in ("samples", "store")]
            differing = sorted({key for trait in traits if trait["base_run"] in bases and base in bases for key in SHARED_FIELDS
                                if bases[trait["base_run"]].get(key) != bases[base].get(key)})
            plan.check(not differing, f"{run_id}: its traits' base runs agree on {', '.join(differing) or 'the samples and store inputs'}")
        held, code, accept_from = {}, {}, None
        if cloud is None:
            made, taken = {"steps": "offline: the base was not read"}, []
        else:
            if (base, tuple(wanted)) not in found:
                held, elsewhere = holders(cloud, base, wanted)
                if elsewhere:
                    plan.notes.append(f"the base {base} names bases in other buckets, not followed: {', '.join(elsewhere)}")
                found[base, tuple(wanted)] = held, {holder: source_of(cloud, holder) for holder in sorted(set(held.values()))}
            held, sources = found[base, tuple(wanted)]
            held = dict(held)
            for holder, source in sources.items():
                if source is None:
                    plan.check(False, f"{run_id}: no staged source of {holder}, which holds {', '.join(step for step in wanted if held.get(step) == holder)}")
                elif source not in downloaded and plan.files:
                    downloaded[source] = cloud.download(source, plan.work / "bases" / str(len(downloaded)))
                    plan.check(downloaded[source] is not None, f"source: svpgs/{source}, {holder}'s, read whole")
            # Which phenotype queries the run makes follows from which of its lists are empty (step_code.left_out).
            lists = run_config(bases, run_id, traits, spec, None) if base in bases and all(trait["base_run"] in bases for trait in traits) else None
            lists = None if lists is None else {key: lists.get(key) or [] for key in ("diseases", "traits")}
            for source in {sources.get(holder) for holder in held.values()}:
                tree = downloaded.get(source)
                if tree is not None and (source, json.dumps(lists)) not in trees:
                    trees[source, json.dumps(lists)] = same_code(commit, tree, lists)
            verdicts = {}
            for step, holder in held.items():
                verdict = trees.get((sources.get(holder), json.dumps(lists)))
                verdicts[step] = None if not verdict else verdict.get(step)
            code = {step: verdict for step, verdict in verdicts.items() if verdict is not None}
            taken, made = reusable(wanted, set(held), code)
            made.update({step: "the run's own traits" for step in spec["reuse_steps"] if step not in wanted})
            if accepts and held.get("store"):
                accept_from = f"{OPS_MOUNT}/{sources.get(held['store'])}" if sources.get(held["store"]) else None
                other = [step for step in taken if sources.get(held[step]) != sources.get(held["store"])]
                if other:
                    plan.check(False, f"{run_id}: accept_steps_from names the store's source, but {', '.join(other)} are held by a run staged "
                               f"from {', '.join(sorted({str(sources.get(held[step])) for step in other}))}: the job remakes any of them whose "
                               "record holds a key of the earlier formula that the store's source does not give", failure="warn")
            if "store" in wanted and "store" not in taken:
                plan.check(False, f"{run_id}: its job would convert the store again ({made['store']}): name a base whose store this commit keeps")
            unchecked = [step for step in wanted if code.get(step) is None and step in held]
            if unchecked:
                plan.check(False, f"{run_id}: the code of {', '.join(unchecked)} is unchecked against the base's source "
                           "(sv_pgs differs and the commit has no launcher/workspace/step_codes.py)")
        config = run_config(bases, run_id, traits, spec, accept_from, warm) if base in bases and all(trait["base_run"] in bases for trait in traits) else None
        if config is not None and run_id == store_id:
            config.update(arm_keys)
        if run_id == store_id:
            shape = dict(STORE_DISK, machine_type=spec["store_machine_type"])
        starts = None
        if warm and cloud is not None:
            kept = kept_store(held["store"]) if "store" in taken else None
            starts = {(trait["name"], arm): warm_start(cloud, trait, arm, kept, saved, keys, metadata, model_format) for trait in traits for arm in arms}
        plan.runs.append({"run_id": run_id, "traits": [trait["name"] for trait in traits], "arms": arms, "base": base, "reuse": taken,
                          "made": {step: why for step, why in made.items() if why != "not asked"}, "config": config, "shape": shape,
                          "holders": {step: held[step] for step in taken}, "starts": starts})
        if spec["reuse"] == "copy":
            for step in taken:
                plan.copies.append((held[step], run_id, step, sum(cloud.listing(f"{held[step]}/run/{step}").values()) if cloud else None))
        reuse_base = base if spec["reuse"] == "base" and wanted else None
        if run_id == store_id:
            through = "fit_store" if "compact_fit_store" in arm_keys else "direct_sv"
            plan.store = _job(run_id, spec["store_slot"], "store", job_environment(spec, run_id, spec["store_slot"], shape, plan.staging, reuse_base, through=through, gpu=False))
            continue
        own = {"PARALLEL_ARMS": "1"} if layout.parallel_arms else {"FIT_ARMS": " ".join(arms)} if layout.merge else {}
        plan.jobs.append(_job(run_id, spec["fit_slot"], "fit", job_environment(spec, run_id, spec["fit_slot"], shape, plan.staging, reuse_base, **own)))
    if warm and cloud is not None:
        # The study's rule (_warm_seeds): a trait's two arms start alike, both cold or each from its own arm's model.
        states: dict = {}
        for run in plan.runs:
            for (trait, _arm), (state, _where) in (run["starts"] or {}).items():
                states.setdefault(trait, set()).add(state)
        unlike = [trait for trait, found in states.items() if len(found) > 1]
        plan.check(not unlike, "warm starts: each trait's arms start alike" + (
            f", but not {', '.join(unlike)}: name the run holding the cold arm's model in its warm_from, or [] for both cold" if unlike else ""))
        starts = [state for run in plan.runs for state, _where in (run["starts"] or {}).values()]
        plan.check("cold" not in starts, f"warm starts: {starts.count('warm')} of {len(starts)} arms start from an earlier model"
                   + (f", {starts.count('cold')} cold (each named under its run)" if "cold" in starts else ""), failure="warn")
    if layout.merge:
        fitted = [run for run in plan.runs if run["run_id"] != store_id]
        groups = [[run for run in fitted if run["traits"] == [trait["name"]]] for trait in spec["traits"]] if layout.runs == "trait_arm" else [fitted]
        jobs = {entry["run_id"]: entry for entry in plan.jobs}
        for group in groups:
            if len(group) < 2:
                continue
            destination, members = group[0], [jobs[run["run_id"]] for run in group]
            trait = destination["traits"][0] if len(destination["traits"]) == 1 else "every trait"
            base = jobs[destination["run_id"]]["environment"].get("BASE_RUN")
            if plan.partner:
                for entry, partner in zip(members, reversed(members)):
                    entry["environment"]["PARTNER_RUN"] = partner["run_id"]
                shape = dict(CPU_MERGE, machine_type=spec["merge_cpu_machine_type"])
                environment = job_environment(spec, destination["run_id"], spec["merge_slot"], shape, plan.staging, base, gpu=False, PARTNER_RUN=group[1]["run_id"])
                plan.finishes.append(dict(_job(destination["run_id"], spec["merge_slot"], "merge", environment, after=[entry["name"] for entry in members]),
                                          runs=[run["run_id"] for run in group], trait=trait, finisher=max(members, key=lambda entry: entry["rate"])))
            else:
                shape = dict(destination["shape"])
                if spec["merge_machine_type"]:
                    shape.update(machine_type=spec["merge_machine_type"], gpu_count=_gpus(spec["merge_machine_type"], shape["gpu_count"]))
                environment = job_environment(spec, destination["run_id"], spec["merge_slot"], shape, plan.staging, base)
                plan.merges.append(dict(_job(destination["run_id"], spec["merge_slot"], "fit", environment, after=[entry["name"] for entry in members]),
                                        destination=destination["run_id"], sources=[(run["run_id"], run["arms"]) for run in group[1:]], trait=trait))
    return plan


def projection(plan: Plan) -> dict:
    """The round's minutes and dollars on the plan's timeline: a fit job's phases (every job at the slowest arm's
    minutes, so the dollars are an upper bound), what follows the slowest arm (its in-job finish, a merge job, or
    nothing), the wall clock without and with one preemption, and the VM and disk dollars of every job but arm_vms'
    store job, which no timeline has minutes for."""
    rows = fit_rows(plan.timeline, plan.spec, plan.arms_in_turn, plan.partner)
    job = (sum(row[1] for row in rows), sum(row[2] for row in rows))
    after = plan.timeline[FINISH[0]] if plan.finishes else plan.timeline[MERGE[0]] if plan.merges else (0, 0)
    preemption = plan.timeline[PREEMPTION[0]]
    fit_usd = tuple(sum(entry["rate"] * minutes / 60 for entry in plan.jobs) for minutes in job)
    merging = [entry["finisher"] for entry in plan.finishes] or plan.merges
    merge_usd = tuple(sum(entry["rate"] * minutes / 60 for entry in merging) for minutes in after)
    wall = (job[0] + after[0], job[1] + after[1])
    return {
        "rows": rows, "job": job, "merge": after, "preemption": preemption, "wall": wall,
        "wall_preempted": (wall[0] + preemption[0], wall[1] + preemption[1]),
        "fit_usd": fit_usd, "merge_usd": merge_usd, "usd": (fit_usd[0] + merge_usd[0], fit_usd[1] + merge_usd[1]),
    }


def _cost(name: str, environment: dict, phases: dict, extra: dict) -> tuple:
    return (name, environment["MACHINE_TYPE"], environment["PROVISIONING_MODEL"], environment["RUN_DISK_TYPE"], int(environment["RUN_DISK_GB"]), phases, extra)


def cost_jobs(plan: Plan) -> list:
    """Each job for ag-cost's round_cost.from_planner: (name, machine, provisioning model, disk type, disk GB, {phase:
    (low, high) minutes}, {"trait", "kind", "after"}). A trait's in-job finish is a merge of its own, minutes on the
    dearer of its arms' machines after both (round_cost.layout_one_gpu's). arm_vms' store job has no minutes on any
    timeline and is left out, and so are the CPU merge jobs, submitted only where no arm's job finished its run."""
    rows = {phase: (low, high) for phase, low, high, _note in fit_rows(plan.timeline, plan.spec, plan.arms_in_turn, plan.partner)}
    traits = {run["run_id"]: run["traits"] for run in plan.runs}
    listed = []
    for entry in plan.jobs:
        trait = traits[entry["run_id"]][0] if len(traits[entry["run_id"]]) == 1 else "every trait"
        listed.append(_cost(entry["name"], entry["environment"], rows, {"trait": trait, "kind": "fit", "after": []}))
    for entry in plan.merges:
        listed.append(_cost(entry["name"], entry["environment"], {MERGE[0]: plan.timeline[MERGE[0]]}, {"trait": entry["trait"], "kind": "merge", "after": entry["after"]}))
    for entry in plan.finishes:
        listed.append(_cost(f"svpgs-{entry['run_id']}-finish", entry["finisher"]["environment"], {MERGE[0]: plan.timeline[FINISH[0]]},
                            {"trait": entry["trait"], "kind": "merge", "after": entry["after"]}))
    return listed


def _shown(environment: dict) -> str:
    return " ".join(f"{name}='{value}'" if " " in value or not value else f"{name}={value}" for name, value in environment.items())


def _span(pair, digits: int = 0) -> str:
    low, high = pair
    shown = f"{low:.{digits}f}", f"{high:.{digits}f}"
    return shown[0] if shown[0] == shown[1] else f"{shown[0]}-{shown[1]}"


def _digits(pair) -> int:
    return 1 if pair[0] % 1 or pair[1] % 1 else 0


def dollars(plan: Plan) -> list:
    """The round's dollars: round_cost.py's projection at the high end of each phase (its expected preemptions, $ by
    phase, and per arm, merge, trait and round against today's layout for the same traits with the target of a fifth),
    with the low end's round; spend.py's prices times the minutes where round_cost.py is not in this checkout."""
    spec = plan.spec
    if round_cost is None:
        projected = projection(plan)
        return [f"dollars, VM and disk at spend.py's {spec['provisioning_model']} prices, every job at the slowest arm's minutes, no preemption: "
                f"fit jobs {_span(projected['fit_usd'])}" + (f", merges {_span(projected['merge_usd'])}" if plan.merges or plan.finishes else "")
                + f"; round {_span(projected['usd'])} (launcher/workspace/round_cost.py is not here)"]
    listed, per_hour = cost_jobs(plan), round_cost.PREEMPTIONS_PER_HOUR
    high, low = (round_cost.project(round_cost.from_planner(listed, end), per_hour) for end in (1, 0))
    today = round_cost.project(round_cost.layout_today([trait["name"] for trait in spec["traits"]], round_cost.MINUTES["today"]), per_hour)
    return [f"dollars (launcher/workspace/round_cost.py, every job at the high end of each phase; at the low end the round is ${low['usd']:.2f}):",
            *("  " + line if line else "" for line in round_cost.report(high, per_hour, today).splitlines())]


def render(plan: Plan) -> str:
    spec, layout = plan.spec, LAYOUTS[plan.spec["layout"]]
    projected = projection(plan)
    shapes = [run["shape"] for run in plan.runs if run["run_id"] != (plan.store or {}).get("run_id")]
    usual = max(shapes, key=shapes.count) if shapes else shape_of(spec, [], [])
    lines = [
        f"round {spec['round']}: {len(spec['traits'])} trait{'s' if len(spec['traits']) > 1 else ''} x {len(spec['arms'])} arms, layout {layout.name} ({layout.summary})",
        f"jobs: {usual['machine_type']} x{usual['gpu_count']} {usual['gpu_type']} {spec['provisioning_model']}, MAX_RETRIES {spec['max_retries']}, "
        f"run disk {usual['run_disk_type']} {usual['run_disk_gb']} GB, through {spec['through']}; steps before the fit by "
        + ("BASE_RUN (no copy)" if spec["reuse"] == "base" else "server-side copy"),
        *plan.notes, "", f"runs ({len(plan.runs)}): each step before the fit kept from the base or remade by the job, and each arm's start:",
        f"  {'RUN_ID':<14} {'arms':<18} {'base':<12} steps before the fit",
    ]
    for run in plan.runs:
        made = ", ".join(f"{step} ({why})" for step, why in run["made"].items()) or "-"
        arms = " ".join(run["arms"]) or "(store job)"
        lines.append(f"  {run['run_id']:<14} {arms:<18} {run['base']:<12} kept: {' '.join(run['reuse']) or '-'}; remade: {made}")
        if len(run["traits"]) > 1 and run is plan.runs[0]:
            lines.append(f"  {'':<14} traits: {' '.join(run['traits'])}")
        if (run["config"] or {}).get("accept_steps_from"):
            lines.append(f"  {'':<14} a record of the earlier key formula kept where {run['config']['accept_steps_from']} made it from these inputs")
        if (run["config"] or {}).get("warm_start_models"):
            lines.append(f"  {'':<14} warm_start_models: {' '.join(run['config']['warm_start_models'])}")
        for (trait, arm), (state, where) in (run["starts"] or {}).items():
            lines.append(f"  {'':<14} {state} {arm} {trait}: {where}")
        if run["starts"] is None and run["arms"] and plan.warm:
            patterns = warm_patterns([trait for trait in spec["traits"] if trait["name"] in run["traits"]])
            lines.append(f"  {'':<14} starts unchecked (the bucket was not read): warm_start_models {' '.join(patterns)}")
        if run["shape"] != usual:
            shape = run["shape"]
            lines.append(f"  {'':<14} shape: {shape['machine_type']}" + (f" x{shape['gpu_count']} {shape['gpu_type']}" if run["run_id"] != (plan.store or {}).get("run_id") else "")
                         + f", {shape['run_disk_type']} {shape['run_disk_gb']} GB")
    if spec["reuse"] == "copy":
        sized = [size for *_names, size in plan.copies if size is not None]
        lines += ["", f"server-side copies: {len(plan.copies)} steps" + (f", {sum(sized) / 1e9:,.0f} GB" if sized else "")
                  + ", each step's objects, then its names and sizes checked against the base's, then its _COMPLETE.json"]
    counts = [f"{len(plan.jobs)} fit"] + ([f"{len(plan.merges)} merge"] if plan.merges else []) + ([f"{len(plan.finishes)} CPU merge where needed"] if plan.finishes else [])
    lines += ["", f"jobs ({'1 store, then ' if plan.store else ''}{', then '.join(counts)}), each by submit.sh with the workspace's settings ("
              + " ".join(WORKSPACE_ENVIRONMENT) + ") from the environment and these:"]
    for entries, what in (([plan.store] if plan.store else [], "store job (submit.sh store)"), (plan.jobs, "fit job (submit.sh fit)"),
                          (plan.merges, "merge job (submit.sh fit)"), (plan.finishes, "CPU merge job (submit.sh merge)")):
        if not entries:
            continue
        common = {name: value for name, value in entries[0]["environment"].items() if all(entry["environment"].get(name) == value for entry in entries)}
        lines.append(f"  every {what}: {_shown(common)}")
        for entry in entries:
            own = {name: value for name, value in entry["environment"].items() if name not in common}
            lines.append(f"  {entry['name']:<26} ${entry['rate']:.2f}/h  {_shown(own)}")
    if plan.store:
        lines.append(f"  the fit jobs wait for {plan.store['name']}: --follow submits them once it has SUCCEEDED")
    if plan.merges:
        lines.append("  each merge job: once its runs' fit jobs have SUCCEEDED with one fit key, the other arm's models go into its run's fit partial and "
                     "--follow submits it")
    if plan.finishes:
        lines.append("  each trait: the later of its two arms' jobs finishes the run on its VM (PARTNER_RUN); --follow submits its CPU merge job only "
                     "where both jobs have ended, neither run holds the last step and each holds its arm's handoff")
    owners = {phase: owner for phase, _what, owner in (*PHASES, MERGE, FINISH, PREEMPTION)}
    budget = {phase: high for phase, _low, high, _note in fit_rows(TIMELINES["budget"], spec, plan.arms_in_turn, plan.partner)}
    budget.update({phase: TIMELINES["budget"][phase][1] for phase in (MERGE[0], FINISH[0])})

    def row(phase: str, pair, note: str = "", label: str = "") -> str:
        over = f"over its budget of {budget[phase]:g}" if phase in budget and pair[1] > budget[phase] else ""
        shown = f"  {label or phase:<14} {_span(pair, _digits(pair)):>11}  {owners.get(phase, ''):<46} {'; '.join(part for part in (note, over) if part)}"
        return shown.rstrip()

    lines += ["", f"minutes per phase, slowest fit job, timeline {plan.timeline_label}:", f"  {'phase':<14} {'minutes':>11}  {'owner':<46} note"]
    lines += [row(phase, (low, high), note) for phase, low, high, note in projected["rows"]]
    lines.append(f"  {'fit job':<14} {_span(projected['job'], _digits(projected['job'])):>11}")
    if plan.finishes:
        lines.append(row(FINISH[0], projected["merge"], "on the later arm's VM, after its fit"))
    elif plan.merges:
        lines.append(row(MERGE[0], projected["merge"], "a merge job after the slowest of its runs"))
    if plan.store:
        lines.append(f"  {'store job':<14} {'?':>11}  {'ag-share':<46} before the fit jobs, once per cohort: the fit store's pass ~30 (ag-cost), "
                     "but measurement and direct_sv made again for the round's cohort, unmeasured; no timeline has its minutes")
    verdict = "met" if projected["wall"][1] <= TARGET_MINUTES else "not met" if projected["wall"][0] > TARGET_MINUTES else "met at the low end only"
    lines.append(f"  {'round':<14} {_span(projected['wall'], _digits(projected['wall'])):>11}  submission to every report"
                 + (" after the store job" if plan.store else "") + f"; goal {TARGET_MINUTES}: {verdict}")
    lines.append(row(PREEMPTION[0], projected["wall_preempted"], "the round with one preemption of the slowest job", label="+1 preemption"))
    lines += ["", *dollars(plan)]
    lines += ["", "checks:"] + [f"  [{state}] {text}" for state, text in plan.checks]
    lines.append("  [--execute] the staging's source.sha256 is the commit's (written before its tree) and its names and sizes are the "
                 "commit's files before any job is submitted, a copied step's "
                 "are its holder's before its _COMPLETE.json, and each job checks its copy of the source against source.sha256")
    failed = [text for state, text in plan.checks if state == "FAIL"]
    lines += ["", "dry run: nothing was changed; " + (f"--execute refuses: {len(failed)} check(s) failed" if failed else "--execute stages, writes, copies and submits the above")]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# --execute and --follow
# ---------------------------------------------------------------------------
def stage(plan: Plan, cloud: Gcloud) -> None:
    """The round's one staging: the commit's tree, its digests and env.tar, each checked against what was meant.

    The digests are written before the tree, so a staging is this commit's exactly when its source.sha256 is: names and
    sizes alone pass a same-length edit, and an interrupted copy of this commit's tree is completed rather than refused."""
    expected = {name: size for name, (size, _digest) in plan.files.items()}
    held = cloud.listing(f"{plan.staging}/source")
    recorded = cloud.read(f"{plan.staging}/source.sha256")
    if (held or recorded is not None) and recorded != manifest(plan.files):
        raise RoundError(f"svpgs/{plan.staging}/source holds another tree: name another round")
    if recorded is None:
        cloud.write(f"{plan.staging}/source.sha256", manifest(plan.files))
    if held != expected:
        cloud.upload(plan.work / "source", plan.staging)
        if cloud.listing(f"{plan.staging}/source") != expected:
            raise RoundError(f"svpgs/{plan.staging}/source is not the commit's tree after the copy; nothing was submitted")
    environment = cloud.size(f"{plan.spec['env_from']}/env.tar")
    if environment is None:
        raise RoundError(f"{plan.spec['env_from']} has no env.tar")
    if cloud.size(f"{plan.staging}/env.tar") != environment:
        cloud.copy(f"{plan.spec['env_from']}/env.tar", f"{plan.staging}/env.tar")
        if cloud.size(f"{plan.staging}/env.tar") != environment:
            raise RoundError(f"env.tar of {plan.spec['env_from']} did not copy whole")


def copy_step(cloud: Gcloud, holder: str, run_id: str, step: str) -> None:
    """One completed step, server-side: its objects but its record, checked name by name and size by size against the
    holder's, then its directories and its record."""
    source, target = f"{holder}/run/{step}", f"{run_id}/run/{step}"
    for _attempt in range(2):
        cloud.copy_tree(source, target, skip=(COMPLETE,))
        wanted = {name: size for name, size in cloud.listing(source).items() if name != COMPLETE}
        if wanted and cloud.listing(target) == wanted:
            break
    else:
        raise RoundError(f"{run_id}: the copy of {holder}'s {step} is not whole after two attempts; its record was not copied")
    if cloud.read(f"{holder}/run/{step}{DIRECTORIES}") is not None:
        cloud.copy(f"{holder}/run/{step}{DIRECTORIES}", f"{run_id}/run/{step}{DIRECTORIES}")
    cloud.copy(f"{source}/{COMPLETE}", f"{target}/{COMPLETE}")


def submit(entry: dict, launcher: Path) -> None:
    """submit.sh with the job's command (fit, store or merge) and environment over the workspace's."""
    missing = [name for name in WORKSPACE_ENVIRONMENT if not os.environ.get(name)]
    if missing:
        raise RoundError(f"the environment lacks the workspace's {missing}")
    done = subprocess.run(["bash", str(launcher / "submit.sh"), entry["command"]], env=dict(os.environ, **entry["environment"]), capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RoundError(f"{entry['name']} was not submitted: {done.stderr.strip()[-500:]}")
    print(f"submitted {entry['name']}", flush=True)


def execute(plan: Plan, cloud: Gcloud, launcher: Path) -> None:
    """The round's staging, each run's config, STAGED_SOURCE object and copied steps, the round's record, and every job
    that waits for no other (arm_vms' store job, or the fit jobs); --follow submits the rest."""
    failed = [text for state, text in plan.checks if state == "FAIL"]
    if failed:
        raise RoundError("refused: " + "; ".join(failed))
    stage(plan, cloud)
    print(f"staged svpgs/{plan.staging}: {len(plan.files)} files, their digests recorded and their names and sizes checked", flush=True)
    for run in plan.runs:
        cloud.write(f"{run['run_id']}/run_config.json", json.dumps(run["config"], indent=2) + "\n")
        cloud.write(f"{run['run_id']}/STAGED_SOURCE", plan.staging + "\n")
        for holder, run_id, step, _size in plan.copies:
            if run_id == run["run_id"]:
                copy_step(cloud, holder, run_id, step)
        print(f"{run['run_id']}: config written" + (f", {' '.join(run['reuse'])} copied and checked" if plan.copies else ""), flush=True)
    record = {"spec": plan.spec, "store": plan.store["name"] if plan.store else None, "jobs": [entry["name"] for entry in plan.jobs],
              "merges": [entry["name"] for entry in plan.merges], "cpu_merges": [entry["name"] for entry in plan.finishes]}
    cloud.write(f"{plan.staging}/round.json", json.dumps(record, indent=2) + "\n")
    for entry in ([plan.store] if plan.store else []) + [entry for entry in plan.jobs if not entry["after"]]:
        submit(entry, launcher)
    if plan.store or plan.merges or plan.finishes:
        print("--follow submits the jobs that wait for these once they have ended", flush=True)


def follow(plan: Plan, cloud: Gcloud, launcher: Path, project: str, region: str) -> list:
    """Each job that waits for others and is not yet submitted, submitted once it may be (with --execute):
    - arm_vms' fit jobs once its store job has SUCCEEDED;
    - a merge job without PARTNER_RUN once its runs' fit jobs have all SUCCEEDED with one fit key, after the other runs'
      models are copied into its run's fit partial (each model's files, then its record);
    - a CPU merge job with PARTNER_RUN only where both fit jobs have ended, neither run holds the round's last step (the
      later arm's job finished neither) and each holds its arm's handoff (effects.partial's handoff.json, written last).
      Its RUN_ID is the run that got further (ag-score's rule): one whose fit is complete or that holds both arms'
      handoffs pushed its adoption, so the merge finishes in it, its partner's models being gone from a completed fit's
      partial; otherwise the plan's order.
    Returns what still waits, with why."""
    states: dict = {}

    def state(name: str):
        if name not in states:
            states[name] = cloud.job_state(name, project, region)
        return states[name]

    waiting = []
    for entry in [entry for entry in plan.jobs if entry["after"]] + plan.merges + plan.finishes:
        names = [f"svpgs-{run}-{plan.spec['merge_slot']}" for run in entry["runs"]] if "finisher" in entry else [entry["name"]]
        if any(state(name) is not None for name in names):
            continue
        before = {name: state(name) or "not found" for name in entry["after"]}
        if "finisher" in entry:
            complete = {run: cloud.completed(run) for run in entry["runs"]}
            if any(plan.spec["through"] in steps for steps in complete.values()):
                continue
            if any(value not in ("SUCCEEDED", "FAILED") for value in before.values()):
                waiting.append((entry["name"], before))
                continue
            handoffs = {run: [name for name in cloud.listing(f"{run}/run/effects.partial/batches") if name.endswith("/handoff.json")] for run in entry["runs"]}
            missing = [run for run, found in handoffs.items() if not found]
            if missing:
                waiting.append((entry["name"], {run: "no handoff: its arm's parts are not mirrored, so resubmit its fit job" for run in missing}))
                continue
            further = [run for run in entry["runs"] if "fit" in complete[run] or len(handoffs[run]) > 1]
            if further and further[0] != entry["run_id"]:
                partner = next(run for run in entry["runs"] if run != further[0])
                entry = dict(entry, run_id=further[0], name=f"svpgs-{further[0]}-{plan.spec['merge_slot']}",
                             environment=dict(entry["environment"], RUN_ID=further[0], PARTNER_RUN=partner))
        elif any(value != "SUCCEEDED" for value in before.values()):
            waiting.append((entry["name"], before))
            continue
        if "sources" in entry:
            runs = [entry["destination"]] + [run for run, _arms in entry["sources"]]
            keys = {run: cloud.read(f"{run}/run/fit.partial/_KEY") for run in runs}
            if None in keys.values() or len(set(keys.values())) != 1:
                waiting.append((entry["name"], {run: "fit key " + ("missing" if key is None else key[:12]) for run, key in keys.items()}))
                continue
        if not cloud.execute:
            waiting.append((entry["name"], {name: f"{value}: ready, --execute submits it" for name, value in before.items()}))
            continue
        for run, arms in entry.get("sources", ()):
            for arm in arms:
                source, target = f"{run}/run/fit.partial/models/{arm}", f"{entry['destination']}/run/fit.partial/models/{arm}"
                records = sorted(name for name in cloud.listing(source) if re.fullmatch(r"[^/]+\.json", name))
                cloud.copy_tree(source, target, skip=records)
                for name in records:
                    cloud.copy(f"{source}/{name}", f"{target}/{name}")
        submit(entry, launcher)
    return waiting


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path, help="The round's spec (ROUND.json).")
    parser.add_argument("--timeline", default="budget", help="budget, today, projected, or a TIMELINE.json of each phase's [low, high] minutes.")
    parser.add_argument("--offline", action="store_true", help="Plan without reading the bucket (its checks are skipped).")
    parser.add_argument("--execute", action="store_true", help="Stage, write each run's config, copy and submit the jobs that wait for none (with --follow: the others).")
    parser.add_argument("--follow", action="store_true", help="Submit each job whose jobs before it have ended as it needs: fit jobs after a store job, merges.")
    parser.add_argument("--watch", type=float, default=0.0, help="With --follow: look again every this many minutes until nothing waits.")
    parser.add_argument("--repository", type=Path, default=HERE.parents[1], help="The git checkout holding the source commit.")
    args = parser.parse_args(argv)
    if args.offline and (args.execute or args.follow):
        parser.error("--execute and --follow read the bucket")
    plan = None
    try:
        spec = read_spec(json.loads(args.spec.read_text(encoding="utf-8")))
        label, timeline = read_timeline(args.timeline)
        bucket = os.environ.get("OPS_BUCKET", "")
        if not args.offline and not bucket:
            raise RoundError("OPS_BUCKET names the workspace's ops bucket (or plan --offline)")
        cloud = None if args.offline else Gcloud(bucket, args.execute)
        plan = build(spec, label, timeline, cloud, args.repository)
        if args.follow:
            project, region = os.environ.get("PROJECT_ID", ""), os.environ.get("REGION", "")
            while True:
                waiting = follow(plan, cloud, HERE, project, region)
                for name, states in waiting:
                    print(f"{name}: " + ", ".join(f"{run} {state}" for run, state in states.items()), flush=True)
                if not waiting or args.watch <= 0:
                    return 0
                time.sleep(args.watch * 60)
        print(render(plan), flush=True)
        if args.execute:
            execute(plan, cloud, HERE)
    except RoundError as error:
        print(f"round: {error}", file=sys.stderr)
        return 2
    finally:
        if plan is not None:
            shutil.rmtree(plan.work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
