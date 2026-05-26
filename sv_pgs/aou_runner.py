"""All of Us orchestration: download VCFs, prepare phenotypes, merge PCs, run one unified fit."""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sv_pgs.all_of_us import (
    DISEASE_DEFINITIONS,
    AllOfUsDiseaseRequest,
    prepare_all_of_us_disease_sample_table,
    resolve_disease_definition,
)
from sv_pgs.gcsfuse_staging import is_gcsfuse_path
from sv_pgs.aou_storage import stage_gcs_object, verify_local_cache
from sv_pgs.path_policy import assert_hot_local_path, assert_safe_for_purpose
from sv_pgs.preflight import (
    assert_preflight_ok,
    check_aou_preflight,
    log_preflight,
)
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.io import load_multi_source_dataset_from_files
from sv_pgs.model import register_fit_checkpoint_path
from sv_pgs.pipeline import run_training_pipeline
from sv_pgs.progress import log, mem
from sv_pgs._typing import NDArray

# Local on-workbench mirror of remote AoU buckets. We download each VCF once
# into work_dir.parent/.sv_pgs_cache/<subdir>/ and reuse it across runs; the
# bcftools precache step then writes an int8 .npy + variants/stats sidecar
# next to each downloaded VCF.
_LOCAL_CACHE_DIRNAME = ".sv_pgs_cache"
_AOU_SV_VCF_CACHE_SUBDIR = "aou_sv_vcfs"
# Microarray PLINK trio lives in its own subdir so that running --variants snp
# alongside --variants sv doesn't interleave files in one place. The trio is
# small enough (~80 GB across .bed + .bim + .fam) that we hold all three files
# locally without per-chromosome sharding.
_AOU_ARRAY_PLINK_CACHE_SUBDIR = "aou_array_plink"
# AoU's microarray PLINK files are emitted with the literal prefix "arrays".
# i.e. arrays.bed / arrays.bim / arrays.fam under .../microarray/plink/.
_AOU_ARRAY_PLINK_PREFIX = "arrays"

# Genotype backend for the downloaded SV cache: "bitpacked" emits a single
# PLINK 1.9 BED trio via ``sv_pgs.sv_transcoder.transcode_sv_vcf_to_bed`` (the
# format consumed by the bitpacked GPU pipeline — see BITPACKED_SPEC.md);
# "int8" keeps the legacy host-side ``*.genotypes.npy`` + sidecar layout. The
# default is bitpacked. Callers can override via ``AOU_GENOTYPE_BACKEND`` in
# the environment without editing this module.
_DEFAULT_GENOTYPE_BACKEND = "bitpacked"


def aou_genotype_backend() -> str:
    """Return the active SV-cache backend ("bitpacked" or "int8")."""
    raw = os.environ.get("AOU_GENOTYPE_BACKEND", _DEFAULT_GENOTYPE_BACKEND)
    value = (raw or _DEFAULT_GENOTYPE_BACKEND).strip().lower()
    if value not in {"bitpacked", "int8"}:
        raise RuntimeError(
            f"AOU_GENOTYPE_BACKEND={raw!r} is invalid; expected 'bitpacked' or 'int8'."
        )
    return value


def _warn_if_work_dir_on_gcsfuse(work_dir: Path) -> None:
    """Surface a loud warning if the local cache dir resolves onto gcsfuse.

    The cache under ``work_dir.parent/.sv_pgs_cache`` is supposed to be on
    local NVMe — staging copies from the gcsfuse-mounted workspace into a
    *gcsfuse* destination would defeat the entire point of the copy. We do
    not silently relocate the cache (the user may have an intentional layout)
    but a runtime warning makes the misconfiguration impossible to miss.
    """
    try:
        if is_gcsfuse_path(work_dir):
            from sv_pgs.progress import log as _log
            _log(
                f"  WARNING: cache work_dir {work_dir} appears to live on a "
                "gcsfuse mount; staged BED/VCF reads will be network-bound. "
                "Pass --cache-dir pointing at local NVMe to avoid this."
            )
    except Exception:
        # Detection is best-effort; never block a run because of it.
        pass

# ---------------------------------------------------------------------------
# AoU paths
#
# Everything lives under CDR_STORAGE_PATH (= gs://fc-aou-datasets-controlled/v8
# at the current Controlled Tier release). The workbench predefines a handful
# of env vars pointing at canonical assets; the source of truth is the
# "Controlled CDR Directory" article in the AoU User Support hub.
#
# Cohorts:
#   srWGS SNP & Indel : 414,830 participants
#   srWGS SVs         :  97,061 participants  (strict subset of the above)
#   genotyping array  : 447,278 participants
# Joint SV+SNP runs are capped at the 97,061 SV samples; the loader's
# _align_sample_ids does the intersection.
#
# srWGS Structural Variants (currently the only source we wire into the
# pipeline). One bgzipped .vcf.gz per autosome.
#   gs://.../v8/wgs/short_read/structural_variants/vcf/full/
#       AoU_srWGS_SV.v8.chr{N}.vcf.gz   (+ .tbi)
#
# srWGS SNP/Indel callsets (available, not yet wired). Same data in five
# formats; pick the one whose shape matches the downstream tool.
#   gs://.../v8/wgs/short_read/snpindel/
#     vds/hail.vds                                  WGS_VDS_PATH
#         Full sparse joint callset (Hail VDS), ~1B sites. Hail-only.
#     acaf_threshold/                               (AF>1% OR AC>100 / ancestry)
#       multiMT/hail.mt                             WGS_ACAF_THRESHOLD_MULTI_HAIL_PATH
#       splitMT/hail.mt                             WGS_ACAF_THRESHOLD_SPLIT_HAIL_PATH
#       vcf/                                        WGS_ACAF_THRESHOLD_VCF_PATH
#           Many .vcf.bgz shards per chromosome — enumerate with `gsutil ls`.
#       plink_bed/  bgen/  pgen/  bed/              (.bed/.bim/.fam etc.)
#         ACAF totals: 57M sites / 116M variants / 414,830 samples.
#     exome/                                        (Gencode v42 exons + 15 bp)
#       multiMT/hail.mt / splitMT/hail.mt           WGS_EXOME_MULTI/SPLIT_HAIL_PATH
#       vcf/                                        WGS_EXOME_VCF_PATH
#       plink_bed/  bgen/  pgen/  bed/
#         Exome totals: 38M sites / 46M variants / 414,830 samples.
#     clinvar/                                      (all ClinVar variants)
#       multiMT/hail.mt / splitMT/hail.mt           WGS_CLINVAR_MULTI/SPLIT_HAIL_PATH
#       vcf/                                        WGS_CLINVAR_VCF_PATH
#       plink_bed/  bgen/  pgen/  bed/
#         ClinVar totals: 1.5M sites / 2.2M variants / 414,830 samples.
#     cmrg/                                         WGS_CMRG_VCF_PATH
#         Challenging Medically Relevant Genes (33 genes) called against a
#         masked hg38 reference. Do NOT intersect with the other callsets.
#
# Auxiliary files we use today (the only path we touch outside the SV VCFs):
#   gs://.../v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv
#       Predicted continental ancestry + 16-dim PCA features per participant.
# Other aux directories that exist but we don't read:
#   vat/ (variant annotations), admixture_estimates/, pgx/ (PharmGKB stars),
#   phasing/, relatedness/ (kinship .tsvs), qc/ (sample QC flags + metrics).
#
# Other CDR roots, for reference:
#   gs://.../v8/wgs/cram/manifest.csv                           WGS_CRAM_MANIFEST_PATH
#   gs://.../v8/microarray/vcf/manifest.csv                     MICROARRAY_VCF_MANIFEST_PATH
#   gs://.../v8/microarray/hail.mt                              MICROARRAY_HAIL_STORAGE_PATH
#   gs://.../v8/microarray/plink/arrays.*                       (PLINK bed for array data)
#   gs://.../v8/microarray/idat/manifest.csv                    MICROARRAY_IDAT_MANIFEST_PATH
#   gs://.../v8/wgs/long_read/manifest.csv                      (lrWGS per-sample manifest)
#   gs://.../v8/known_issues/                                   (issue-specific sample lists)
# ---------------------------------------------------------------------------

# The two env vars below are interpolated into gsutil command arguments
# (CDR_STORAGE_PATH builds the `gs://.../...` source URI; GOOGLE_PROJECT is
# passed via `-u <project>`). subprocess uses list-form invocation so a shell
# can't reinterpret embedded metacharacters, but a value starting with '-'
# would be parsed by gsutil itself as an option flag (argument injection).
# Reject anything that looks suspicious so a typo or hostile env never turns
# into a command-line option.
import re as _re

_GSUTIL_PROJECT_RE = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.:]*$")
_GSUTIL_GS_URI_RE = _re.compile(r"^gs://[A-Za-z0-9_\-][A-Za-z0-9_\-./]*$")


def _cdr_storage_path() -> str:
    # gs://fc-aou-datasets-controlled/v8 on the current Controlled Tier; the
    # workbench sets this env var for every notebook + container.
    value = os.environ.get("CDR_STORAGE_PATH")
    if not value:
        raise RuntimeError("CDR_STORAGE_PATH is not set. Are you on an All of Us workbench?")
    stripped = value.strip()
    if not _GSUTIL_GS_URI_RE.match(stripped):
        raise RuntimeError(
            f"Refusing to use CDR_STORAGE_PATH={stripped!r}: must be a gs:// URI "
            "composed of alphanumerics, underscore, hyphen, dot, and slash."
        )
    return stripped


def _google_project() -> str:
    # Required as the gsutil -u billing project for any CDR egress (AoU does
    # not absorb bucket-read costs; the call fails without it).
    value = os.environ.get("GOOGLE_PROJECT")
    if not value:
        raise RuntimeError("GOOGLE_PROJECT is not set. Are you on an All of Us workbench?")
    stripped = value.strip()
    if not _GSUTIL_PROJECT_RE.match(stripped):
        raise RuntimeError(
            f"Refusing to use GOOGLE_PROJECT={stripped!r}: GCP project IDs may "
            "start with an alphanumeric and contain only letters, digits, "
            "underscore, hyphen, dot, or colon."
        )
    return stripped


def sv_vcf_dir() -> str:
    # SV VCFs live under structural_variants/vcf/full — one .vcf.gz per
    # autosome (no env var; the path is documented in the Controlled CDR
    # Directory but not exposed via WGS_*_VCF_PATH).
    return f"{_cdr_storage_path()}/wgs/short_read/structural_variants/vcf/full"


# Candidate basenames for the ancestry predictions file. AoU has shipped this
# under both names across CDR releases — newer drops prefix it with the model
# version (e.g. echo_v4_r2.) while older drops used the bare name.
_ANCESTRY_PREDS_BASENAMES = (
    "echo_v4_r2.ancestry_preds.tsv",
    "ancestry_preds.tsv",
)


def _ancestry_aux_dir() -> str:
    return f"{_cdr_storage_path()}/wgs/short_read/snpindel/aux/ancestry"


def _gsutil_object_exists(path: str) -> bool:
    if not _GSUTIL_GS_URI_RE.match(path):
        return False
    cmd = ["gsutil", "-u", _google_project(), "-q", "stat", path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def resolve_ancestry_predictions_path() -> str:
    # Per-participant ancestry predictions + PC features (.tsv keyed by
    # research_id). Same file is reused as the source of the 16 covariate PCs
    # the pipeline merges into the sample table. Probe known basenames so a
    # bucket-naming change (echo_v4_r2.* vs. bare) doesn't break the run.
    aux_dir = _ancestry_aux_dir()
    for basename in _ANCESTRY_PREDS_BASENAMES:
        candidate = f"{aux_dir}/{basename}"
        if _gsutil_object_exists(candidate):
            return candidate
    raise RuntimeError(
        "Could not locate ancestry_preds.tsv under "
        f"{aux_dir} (tried: {', '.join(_ANCESTRY_PREDS_BASENAMES)})."
    )


def local_ancestry_predictions_path(work_dir: Path) -> Path:
    # Park the ancestry predictions one level above work_dir (alongside the
    # SV VCF and microarray PLINK caches) so an all-diseases sweep shares the
    # single ~3 GB download across every disease's run instead of refetching
    # it into each disease-specific work_dir.
    return work_dir.parent / _LOCAL_CACHE_DIRNAME / "aou_ancestry" / "ancestry_preds.tsv"


def sv_vcf_name(chromosome: int) -> str:
    # AoU's canonical SV VCF filename, hard-coded against the v8 release.
    # Bump the version literal when AoU rolls a new SV callset.
    return f"AoU_srWGS_SV.v8.chr{chromosome}.vcf.gz"


def local_sv_vcf_cache_dir(work_dir: Path) -> Path:
    # We park the SV VCF mirror one level above work_dir so multiple disease
    # runs (each with its own work_dir) share the same downloaded VCFs.
    return work_dir.parent / _LOCAL_CACHE_DIRNAME / _AOU_SV_VCF_CACHE_SUBDIR


def local_sv_vcf_path(chromosome: int, work_dir: Path) -> Path:
    return local_sv_vcf_cache_dir(work_dir) / sv_vcf_name(chromosome)


def array_plink_dir() -> str:
    # AoU's microarray PLINK 1 trio lives at .../microarray/plink/arrays.{bed,bim,fam}.
    # Single set (NOT chromosome-sharded), ~447,278 samples × ~700k SNPs,
    # lifted over to hg38 from the original genotyping-array calls. Used here
    # as the SNP source for joint SV+SNP runs because the file shape matches
    # PlinkRawGenotypeMatrix natively and the total size (~80 GB) fits a
    # standard workbench disk — unlike the ACAF SNP/indel callset at ~12 TB.
    return f"{_cdr_storage_path()}/microarray/plink"


def local_array_plink_cache_dir(work_dir: Path) -> Path:
    return work_dir.parent / _LOCAL_CACHE_DIRNAME / _AOU_ARRAY_PLINK_CACHE_SUBDIR


def local_array_plink_path(work_dir: Path) -> Path:
    # Returns the .bed path; the sibling .bim and .fam live in the same dir.
    return local_array_plink_cache_dir(work_dir) / f"{_AOU_ARRAY_PLINK_PREFIX}.bed"


def _mounted_cdr_roots() -> list[Path]:
    roots = [Path.home() / "workspace" / "vwb-aou-datasets-controlled" / "v8"]
    return [root for root in roots if root.exists()]


def _mounted_cdr_file(relative_path: str) -> Path | None:
    for root in _mounted_cdr_roots():
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    return None


def _link_mounted_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            if target.samefile(source):
                return
        except OSError:
            pass
        return
    target.unlink(missing_ok=True)
    target.symlink_to(source)


def _mounted_array_plink_prefix() -> Path | None:
    for root in _mounted_cdr_roots():
        prefix = root / "microarray" / "plink" / _AOU_ARRAY_PLINK_PREFIX
        if all(
            (prefix.parent / f"{prefix.name}.{ext}").exists()
            for ext in ("bed", "bim", "fam")
        ):
            return prefix
    return None


def _link_mounted_array_plink(work_dir: Path) -> bool:
    mounted_prefix = _mounted_array_plink_prefix()
    if mounted_prefix is None:
        return False
    cache_dir = local_array_plink_cache_dir(work_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _warn_if_work_dir_on_gcsfuse(cache_dir)
    bed_source = mounted_prefix.parent / f"{mounted_prefix.name}.bed"
    bed_target = cache_dir / f"{_AOU_ARRAY_PLINK_PREFIX}.bed"
    # Replace any pre-existing symlink into the mount; gcsfuse-backed
    # symlinks turn every random read into an HTTP GET against GCS.
    for extension in ("bed", "bim", "fam"):
        candidate = cache_dir / f"{_AOU_ARRAY_PLINK_PREFIX}.{extension}"
        if candidate.is_symlink():
            try:
                if is_gcsfuse_path(candidate.resolve()):
                    candidate.unlink()
            except OSError:
                candidate.unlink(missing_ok=True)
    # Symlink the trio from the cache dir into the gcsfuse-mounted source.
    # The .bed will be read ONCE, sequentially, by the bitpacked_loader (which
    # accepts a gcsfuse path via assert_safe_for_purpose(..., allow_sequential
    # _gcsfuse=True) and uses POSIX_FADV_SEQUENTIAL + large chunked reads).
    # Forcing a ~194 GB hard copy here costs 90+ minutes at ~36 MB/s, which is
    # strictly worse than streaming once at ~200 MB/s straight into HBM.
    # The .bim/.fam siblings are tiny but get re-read by downstream code; that
    # is still fine over gcsfuse because they are small and cached.
    if is_gcsfuse_path(bed_source):
        log(
            "  microarray: symlinking PLINK trio from gcsfuse workspace "
            "(BED streams directly into HBM via the bitpacked loader)"
        )
    for extension in ("bim", "fam"):
        source = mounted_prefix.parent / f"{mounted_prefix.name}.{extension}"
        target = cache_dir / f"{_AOU_ARRAY_PLINK_PREFIX}.{extension}"
        _link_mounted_file(source, target)
    _link_mounted_file(bed_source, bed_target)
    return True


def _link_mounted_sv_vcf(chromosome: int, work_dir: Path) -> bool:
    name = sv_vcf_name(chromosome)
    relative_vcf = f"wgs/short_read/structural_variants/vcf/full/{name}"
    mounted_vcf = _mounted_cdr_file(relative_vcf)
    mounted_tbi = _mounted_cdr_file(f"{relative_vcf}.tbi")
    if mounted_vcf is None or mounted_tbi is None:
        return False
    local_vcf = local_sv_vcf_path(chromosome, work_dir)
    local_tbi = local_vcf.parent / f"{name}.tbi"
    _warn_if_work_dir_on_gcsfuse(local_vcf.parent)
    # If a prior run symlinked these into gcsfuse, drop the links so we can
    # restage as a hard local copy. Symlinks pointed at local files are kept.
    for candidate in (local_vcf, local_tbi):
        if candidate.is_symlink():
            try:
                if is_gcsfuse_path(candidate.resolve()):
                    candidate.unlink()
            except OSError:
                candidate.unlink(missing_ok=True)
    if is_gcsfuse_path(mounted_vcf):
        # Hard COPY both VCF and index to local NVMe; mmap/tabix random reads
        # over gcsfuse turn into per-page HTTP GETs.
        required_bytes = mounted_vcf.stat().st_size + mounted_tbi.stat().st_size
        _check_disk_space(local_vcf.parent, required_bytes)
        log(
            f"  chr{chromosome}: copying SV VCF + index from gcsfuse workspace "
            f"to local cache ({required_bytes / 1e9:.2f} GB)"
        )
        # Manifest-emitting atomic stage (aou_storage). The .tbi gets a
        # distinct content_kind so cache invalidation can target it alone.
        stage_gcs_object(str(mounted_vcf), local_vcf, content_kind="aou_sv_vcf")
        stage_gcs_object(str(mounted_tbi), local_tbi, content_kind="aou_sv_vcf_tbi")
        return True
    _link_mounted_file(mounted_vcf, local_vcf)
    _link_mounted_file(mounted_tbi, local_tbi)
    return True


def _link_mounted_ancestry_preds(work_dir: Path) -> bool:
    mounted = None
    for basename in _ANCESTRY_PREDS_BASENAMES:
        mounted = _mounted_cdr_file(
            f"wgs/short_read/snpindel/aux/ancestry/{basename}"
        )
        if mounted is not None:
            break
    if mounted is None:
        return False
    _link_mounted_file(mounted, local_ancestry_predictions_path(work_dir))
    return True


# ---------------------------------------------------------------------------
# gsutil helpers
# ---------------------------------------------------------------------------

def _check_disk_space(path: Path, required_bytes: int) -> None:
    """Raise if the filesystem doesn't have enough free space."""
    stat = shutil.disk_usage(str(path))
    if stat.free < required_bytes:
        free_gb = stat.free / 1e9
        need_gb = required_bytes / 1e9
        raise RuntimeError(
            f"Not enough disk space: {free_gb:.1f} GB free, need {need_gb:.1f} GB "
            f"at {path}"
        )


def _gsutil_cp(src: str, dst: str) -> None:
    """Download with gsutil, showing real-time progress via -m flag."""
    # Defense in depth: even though `src` is always built from our own constants
    # plus a validated CDR_STORAGE_PATH, refuse a value that gsutil would
    # interpret as an option flag (anything starting with '-'). The remote URI
    # must be a gs:// URI; reject the unlikely but dangerous case where it
    # somehow isn't.
    if not src.startswith("gs://") or src.startswith("-"):
        raise RuntimeError(f"Refusing to gsutil cp from non-gs:// source: {src!r}")
    if dst.startswith("-"):
        raise RuntimeError(f"Refusing to gsutil cp to suspicious destination: {dst!r}")
    cmd = ["gsutil", "-u", _google_project(), "-m", "cp", src, dst]
    log(f"  downloading {src}")
    # Stream output in real time so user sees progress.
    # ``with`` ensures the process and its stdout pipe are reaped even when an
    # exception (e.g. KeyboardInterrupt, or a logging failure mid-iteration)
    # interrupts the read loop — otherwise the gsutil child would orphan and
    # the read end of the pipe would leak a file descriptor.
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    stripped = line.strip()
                    if stripped:
                        log(f"    {stripped}")
            process.wait()
        except BaseException:
            # Kill the child so we don't leave a long-running gsutil orphaned.
            try:
                process.kill()
            except OSError:
                pass
            raise
    if process.returncode != 0:
        raise RuntimeError(f"gsutil cp failed (exit {process.returncode}): {src}")


def _gsutil_size(path: str) -> int:
    """Get the size of a remote GCS object in bytes."""
    if not path.startswith("gs://") or path.startswith("-"):
        raise RuntimeError(f"Refusing to gsutil du on non-gs:// path: {path!r}")
    cmd = ["gsutil", "-u", _google_project(), "du", "-s", path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw_first = result.stdout.strip().split()
    if not raw_first:
        raise RuntimeError(f"gsutil du returned empty output for {path}")
    try:
        return int(raw_first[0])
    except ValueError as exc:
        raise RuntimeError(
            f"gsutil du returned non-integer size for {path}: {raw_first[0]!r}"
        ) from exc


def _download_gcs_object_if_missing(remote_path: str, local_path: Path) -> None:
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # Use a per-process, per-call unique partial name so concurrent disease
    # sweeps (multiple PIDs racing to fetch the same shared file) never share
    # the same staging path. The losing racer's atomic replace still wins
    # safely; the local_path.exists() recheck below short-circuits redundant
    # downloads on subsequent calls.
    partial_path = local_path.with_name(
        f"{local_path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        _gsutil_cp(remote_path, str(partial_path))
        if local_path.exists():
            # Another concurrent fetcher won the race and atomically published
            # the final file. Discard our staging copy.
            partial_path.unlink(missing_ok=True)
            return
        # os.replace (Path.replace) is atomic on POSIX and overwrites cleanly.
        partial_path.replace(local_path)
    except (OSError, subprocess.SubprocessError, RuntimeError):
        partial_path.unlink(missing_ok=True)
        raise


def download_sv_vcf(chromosome: int, work_dir: Path) -> Path:
    """Download one SV VCF + index when needed and return the local VCF path."""
    name = sv_vcf_name(chromosome)
    local_vcf = local_sv_vcf_path(chromosome, work_dir)
    local_vcf.parent.mkdir(parents=True, exist_ok=True)
    local_tbi = local_vcf.parent / f"{name}.tbi"
    # ``verify_local_cache`` checks file presence AND a sibling manifest with
    # ``complete=True`` + matching size, so a torn copy from a killed run no
    # longer masquerades as a valid cache entry.
    vcf_cached = verify_local_cache(local_vcf)
    tbi_cached = verify_local_cache(local_tbi)
    if (not vcf_cached or not tbi_cached) and _link_mounted_sv_vcf(chromosome, work_dir):
        log(f"  chr{chromosome}: linked VCF + index from mounted workspace resource")
        vcf_cached = verify_local_cache(local_vcf)
        tbi_cached = verify_local_cache(local_tbi)

    if vcf_cached and tbi_cached:
        log(f"  chr{chromosome}: VCF already present in local cache")
        assert_hot_local_path(local_vcf, purpose="aou_sv_vcf_read")
        return local_vcf

    remote_dir = sv_vcf_dir()
    vcf_remote = f"{remote_dir}/{name}"
    tbi_remote = f"{remote_dir}/{name}.tbi"
    missing_downloads = []
    if not vcf_cached:
        missing_downloads.append((vcf_remote, local_vcf, "VCF"))
    if not tbi_cached:
        missing_downloads.append((tbi_remote, local_tbi, "index"))

    required_bytes = sum(_gsutil_size(remote) for remote, _, _ in missing_downloads)
    cache_dir = local_sv_vcf_cache_dir(work_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _check_disk_space(cache_dir, required_bytes)
    missing_labels = " + ".join(label for _, _, label in missing_downloads)
    log(
        f"  chr{chromosome}: downloading missing {missing_labels} into cache "
        + f"{cache_dir} ({required_bytes/1e9:.1f} GB)"
    )
    for remote, local_path, _ in missing_downloads:
        _download_gcs_object_if_missing(remote, local_path)
    assert_hot_local_path(local_vcf, purpose="aou_sv_vcf_read")
    return local_vcf


def _is_valid_cache_entry(path: Path) -> bool:
    """True if ``path`` is a usable cache entry.

    A symlink whose target is a regular file counts as valid even when no
    ``*.manifest.json`` sibling exists: manifests are emitted by
    ``stage_gcs_object`` (a COPY into local cache), not by ``_link_mounted_*``
    paths which symlink into the workspace mount. Gating symlinks on
    ``verify_local_cache`` would force a 137 GB re-stage of a file we already
    have a valid pointer to.
    """
    try:
        if path.is_symlink():
            try:
                target = path.resolve()
            except (OSError, RuntimeError):
                return False
            return target.is_file()
    except OSError:
        return False
    return verify_local_cache(path)


def _cleanup_stale_partials(cache_dir: Path, prefix: str) -> None:
    """Unlink ``<prefix>*.partial.*`` and ``<prefix>*.tmp`` leftovers.

    A run killed mid-copy leaves behind ``arrays.bed.partial.<pid>.<uuid>``
    files (and matching ``.tmp`` siblings from the legacy gcsfuse_staging
    path). They are never resumed — the next stage_gcs_object call writes a
    fresh partial with its own pid+uuid suffix — so anything left over from
    a dead previous run is pure dead weight (137 GB in the worst case).
    """
    if not cache_dir.exists():
        return
    patterns = (f"{prefix}*.partial.*", f"{prefix}*.tmp")
    for pattern in patterns:
        for stale in cache_dir.glob(pattern):
            try:
                if stale.is_file() or stale.is_symlink():
                    log(f"  cleaning stale partial: {stale}")
                    stale.unlink()
            except OSError:
                pass


def download_array_plink(work_dir: Path) -> Path:
    """Download the AoU microarray PLINK 1 trio and return the local .bed path.

    The trio is a single file set (no chromosome sharding) covering all 447k
    array participants. After the first run the files stay on disk under
    work_dir.parent/.sv_pgs_cache/aou_array_plink/; subsequent runs reuse them.
    """
    cache_dir = local_array_plink_cache_dir(work_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any *.partial.* / *.tmp from a killed previous run before we
    # decide whether the cache is valid; otherwise verify_local_cache may
    # treat a stale 137 GB partial as evidence the cache is in flux.
    _cleanup_stale_partials(cache_dir, _AOU_ARRAY_PLINK_PREFIX)
    local_bed = local_array_plink_path(work_dir)
    if _link_mounted_array_plink(work_dir):
        if all(
            _is_valid_cache_entry(cache_dir / f"{_AOU_ARRAY_PLINK_PREFIX}.{ext}")
            for ext in ("bed", "bim", "fam")
        ):
            log("  microarray: linked PLINK trio from mounted workspace resource")
            assert_safe_for_purpose(
                local_bed,
                purpose="aou_array_plink_bed_read",
                allow_sequential_gcsfuse=True,
            )
            return local_bed

    missing_extensions = []
    for extension in ("bed", "bim", "fam"):
        local_path = cache_dir / f"{_AOU_ARRAY_PLINK_PREFIX}.{extension}"
        if not _is_valid_cache_entry(local_path):
            missing_extensions.append(extension)

    if not missing_extensions:
        log("  microarray: PLINK trio already present in local cache")
        assert_safe_for_purpose(
            local_bed,
            purpose="aou_array_plink_bed_read",
            allow_sequential_gcsfuse=True,
        )
        return local_bed

    remote_dir = array_plink_dir()
    missing_downloads: list[tuple[str, Path, str]] = []
    for extension in missing_extensions:
        local_path = cache_dir / f"{_AOU_ARRAY_PLINK_PREFIX}.{extension}"
        remote_path = f"{remote_dir}/{_AOU_ARRAY_PLINK_PREFIX}.{extension}"
        missing_downloads.append((remote_path, local_path, extension))

    required_bytes = sum(_gsutil_size(remote) for remote, _, _ in missing_downloads)
    _check_disk_space(cache_dir, required_bytes)
    missing_labels = " + ".join(label for _, _, label in missing_downloads)
    log(
        f"  microarray: downloading missing {missing_labels} into cache "
        + f"{cache_dir} ({required_bytes/1e9:.1f} GB)"
    )
    for remote, local_path, _ in missing_downloads:
        _download_gcs_object_if_missing(remote, local_path)
    assert_safe_for_purpose(
        local_bed,
        purpose="aou_array_plink_bed_read",
        allow_sequential_gcsfuse=True,
    )
    return local_bed


# ---------------------------------------------------------------------------
# Ancestry / PC merging
# ---------------------------------------------------------------------------

def download_ancestry_preds(work_dir: Path) -> Path:
    """Download the AoU ancestry predictions file (contains per-sample PCs)."""
    local = local_ancestry_predictions_path(work_dir)
    if local.exists():
        log(f"  ancestry predictions already present: {local.name}")
        return local
    # Migrate from the previous per-disease location (work_dir/ancestry_preds.tsv)
    # so existing runs don't have to re-download into the new shared cache.
    legacy_local = work_dir / "ancestry_preds.tsv"
    if legacy_local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            legacy_local.replace(local)
            log(f"  migrated legacy ancestry cache: {legacy_local} -> {local}")
            return local
        except OSError:
            pass
    if _link_mounted_ancestry_preds(work_dir):
        log(f"  linked ancestry predictions from mounted workspace resource: {local.name}")
        return local
    remote = resolve_ancestry_predictions_path()
    log(f"  downloading ancestry predictions: {remote}")
    # Route through the temp-and-replace helper so an interrupted/failed
    # gsutil cp leaves a `.partial` file we discard, rather than a truncated
    # `ancestry_preds.tsv` that future runs would silently reuse.
    _download_gcs_object_if_missing(remote, local)
    return local

def release_process_memory() -> None:
    gc.collect()
    try:
        import jax

        clear_caches = getattr(jax, "clear_caches", None)
        if callable(clear_caches):
            clear_caches()
    except (ImportError, RuntimeError):
        pass
    try:
        import cupy as cp

        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except (ImportError, OSError, RuntimeError):
        pass


def _parse_pca_features_column(series: pd.Series, n_pcs: int) -> tuple[pd.DataFrame, list[str]]:
    """Parse a compound pca_features column like '[0.01,-0.02,...]' into PC1..PCn."""
    pc_names = [f"PC{i+1}" for i in range(n_pcs)]
    n_failed = 0

    # AoU PCA-feature strings are short (e.g. 16 floats ≈ 250 bytes). Cap the
    # per-row length so a malformed ancestry file with a multi-megabyte JSON
    # blob in one row can't blow up worker memory while we parse the table.
    _PCA_FEATURES_MAX_LEN = 8 * 1024

    def _parse_row(val: str) -> list[float]:
        nonlocal n_failed
        if not isinstance(val, str) or not val.strip():
            n_failed += 1
            return [float("nan")] * n_pcs
        cleaned = val.strip()
        if len(cleaned) > _PCA_FEATURES_MAX_LEN:
            n_failed += 1
            return [float("nan")] * n_pcs
        try:
            if cleaned.startswith("["):
                values = json.loads(cleaned)
                # Reject anything that isn't a flat list of scalars; a nested
                # structure (or a dict) would otherwise propagate weird types
                # straight into the downstream float() conversion or fail with
                # an unrelated TypeError.
                if not isinstance(values, list) or len(values) > 4 * n_pcs:
                    raise ValueError("pca_features must be a flat list of numbers")
            else:
                values = [float(x.strip()) for x in cleaned.split(",") if x.strip()]
            return [float(values[i]) if i < len(values) else float("nan") for i in range(n_pcs)]
        except (json.JSONDecodeError, ValueError, TypeError, IndexError):
            n_failed += 1
            return [float("nan")] * n_pcs

    # Log the format of the first non-empty value (truncated, no individual data)
    first_val = series.dropna().iloc[0] if len(series.dropna()) > 0 else ""
    preview = str(first_val)[:80] + ("..." if len(str(first_val)) > 80 else "")
    log(f"  pca_features format preview (first row, truncated): {preview}")

    parsed = series.apply(_parse_row)
    df = pd.DataFrame(parsed.tolist(), columns=pd.Index(pc_names), index=series.index)

    n_total = len(series)
    n_parsed = n_total - n_failed
    log(f"  parsed: {n_parsed}/{n_total} rows successfully ({n_failed} failed)")
    if n_parsed > 0:
        # Log aggregate stats only (no individual values)
        for col in pc_names[:3]:
            vals = df[col].dropna()
            if len(vals) > 0:
                log(f"    {col}: mean={vals.mean():.4f}  std={vals.std():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

    return df, pc_names


def merge_pcs_into_sample_table(
    sample_table_path: Path,
    ancestry_path: Path,
    output_path: Path,
    n_pcs: int = 10,
) -> tuple[Path, list[str]]:
    """Merge top N PCs from ancestry file into the sample table. Returns (output_path, pc_column_names)."""
    if output_path.exists():
        existing = pd.read_csv(output_path, sep="\t", nrows=0)
        existing_pc_cols = sorted(
            [c for c in existing.columns if c.startswith("PC") and c[2:].isdigit()],
            key=lambda x: int(x[2:]),
        )
        merged_mtime = output_path.stat().st_mtime
        inputs_mtime = max(sample_table_path.stat().st_mtime, ancestry_path.stat().st_mtime)
        if merged_mtime > inputs_mtime and len(existing_pc_cols) == int(n_pcs):
            log("  PC-merged table up to date, skipping recomputation")
            return output_path, existing_pc_cols
        log(f"  merged sample table already exists but is stale; recomputing with n_pcs={n_pcs}")

    log(f"  loading sample table: {sample_table_path}")
    samples = pd.read_csv(sample_table_path, sep="\t", dtype={"sample_id": str, "person_id": str})

    log(f"  loading ancestry predictions: {ancestry_path}")
    ancestry = pd.read_csv(ancestry_path, sep="\t", dtype=str)
    log(f"  ancestry columns ({len(ancestry.columns)}): {list(ancestry.columns)}")
    log(f"  ancestry rows: {len(ancestry)}, sample table rows: {len(samples)}")

    # Find the ID column in ancestry
    id_col = None
    for candidate in ["research_id", "person_id", "sample_id", "s"]:
        if candidate in ancestry.columns:
            id_col = candidate
            break
    if id_col is None:
        raise RuntimeError(f"No ID column found in ancestry file. Columns: {list(ancestry.columns)}")

    # Extract PCs: either individual columns (PC1, PC2, ...) or a compound pca_features column
    if "pca_features" in ancestry.columns:
        log(f"  parsing compound pca_features column into PC1..PC{n_pcs}")
        pca_features_col = ancestry["pca_features"]
        if not isinstance(pca_features_col, pd.Series):
            raise RuntimeError("Expected 'pca_features' to be a single column, got a DataFrame")
        pc_df, pc_cols = _parse_pca_features_column(pca_features_col, n_pcs)
        for col in pc_cols:
            ancestry[col] = pc_df[col].values
    else:
        pc_cols = sorted(
            [c for c in ancestry.columns if c.startswith("PC") and c[2:].isdigit()],
            key=lambda x: int(x[2:]),
        )[:n_pcs]
        if not pc_cols:
            raise RuntimeError(f"No PC columns found in ancestry file. Columns: {list(ancestry.columns)}")
        for col in pc_cols:
            ancestry[col] = pd.to_numeric(ancestry[col], errors="coerce")

    log(f"  extracted {len(pc_cols)} PCs: {pc_cols}")

    # Add age^2
    if "age_at_observation_start" in samples.columns:
        samples["age_at_observation_start"] = pd.to_numeric(samples["age_at_observation_start"], errors="coerce")
        samples["age_squared"] = samples["age_at_observation_start"] ** 2
        log("  added age_squared covariate")

    # Check ID overlap before merging (aggregate counts only, no individual IDs)
    ancestry_ids = set(ancestry[id_col].dropna().astype(str))
    sample_id_sets = {
        column: set(samples[column].dropna().astype(str))
        for column in ("person_id", "sample_id")
        if column in samples.columns
    }
    if not sample_id_sets:
        raise RuntimeError("Sample table must contain at least one of: person_id, sample_id")

    overlap_counts = {column: len(ancestry_ids & ids) for column, ids in sample_id_sets.items()}
    log(f"  ancestry ID column: {id_col} ({len(ancestry_ids)} unique IDs)")
    for column in ("person_id", "sample_id"):
        if column in sample_id_sets:
            log(f"  overlap with {column}: {overlap_counts[column]}/{len(sample_id_sets[column])}")

    merge_key = max(overlap_counts, key=lambda column: overlap_counts[column])
    if overlap_counts[merge_key] == 0:
        available_keys = ", ".join(sample_id_sets)
        raise RuntimeError(
            f"No ID overlap between ancestry column '{id_col}' and sample table columns: {available_keys}"
        )
    log(f"  merging on {merge_key} ({overlap_counts[merge_key]} matches)")

    ancestry_subset = ancestry[[id_col] + pc_cols].copy()
    new_columns = [merge_key if column == id_col else column for column in ancestry_subset.columns]
    ancestry_subset.columns = pd.Index(new_columns)
    merged = samples.merge(ancestry_subset, on=merge_key, how="left")
    n_with_pcs = int(merged[pc_cols[0]].notna().sum())

    log(f"  merged: {len(merged)} rows, {n_with_pcs} with PCs ({100*n_with_pcs/max(len(merged),1):.0f}%)")
    if n_with_pcs == 0:
        raise RuntimeError(
            "Ancestry merge produced zero rows with PCs. "
            + f"merge_key={merge_key} ancestry_path={ancestry_path} sample_table_path={sample_table_path}"
        )

    merged.to_csv(output_path, sep="\t", index=False)
    return output_path, pc_cols


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def _aou_run_metadata_path(work_dir: Path) -> Path:
    return work_dir / "aou_run_metadata.json"


def _validate_aou_chromosomes(chromosomes: list[int]) -> list[int]:
    if not chromosomes:
        raise ValueError("chromosomes cannot be empty.")
    normalized = [int(chromosome) for chromosome in chromosomes]
    invalid = [chromosome for chromosome in normalized if chromosome < 1 or chromosome > 22]
    if invalid:
        raise ValueError(f"chromosomes must be autosomes 1-22; got {invalid}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"chromosomes must be unique; got {normalized}")
    return normalized


def _build_aou_run_metadata(
    *,
    disease: str,
    chromosomes: list[int],
    n_pcs: int,
    pc_cols: list[str],
    covariates: list[str],
    max_outer_iterations: int,
    random_seed: int,
    variant_metadata_path: str | None,
    variants: str,
    beta_variance_update_interval: int,
    final_posterior_diagnostics: bool,
    validation_interval: int,
    validate_first_iteration: bool,
    sample_space_preconditioner_rank: int,
    test_fraction: float = 0.0,
    marginal_screen_min_abs_z: float = 0.0,
) -> dict[str, object]:
    return {
        "disease": disease,
        "chromosomes": chromosomes,
        "requested_n_pcs": n_pcs,
        "effective_pc_columns": pc_cols,
        "covariates": covariates,
        "max_outer_iterations": max_outer_iterations,
        "random_seed": random_seed,
        "variant_metadata_path": variant_metadata_path,
        # Different --variants choices fit different models. Including this
        # in the run metadata makes existing-result-reuse skip a cached fit
        # when the source mix changes.
        "variants": variants,
        "beta_variance_update_interval": int(beta_variance_update_interval),
        "final_posterior_diagnostics": bool(final_posterior_diagnostics),
        "validation_interval": int(validation_interval),
        "validate_first_iteration": bool(validate_first_iteration),
        "sample_space_preconditioner_rank": int(sample_space_preconditioner_rank),
        # Float so resuming a 0.2-test run picks up the cached fit but a flip
        # to 0.1 triggers a re-fit. Stored at 6-digit precision to keep the
        # JSON canonical across host float formatting differences.
        "test_fraction": round(float(test_fraction), 6),
        # |z| pre-screen threshold changes the active-variant set; if a user
        # tightens or loosens the screen, the cached fit no longer matches.
        "marginal_screen_min_abs_z": round(float(marginal_screen_min_abs_z), 6),
    }


# Canonical --variants choices. The CLI exposes a few aliases on top of these
# ("snv" for "snp", "sv+snp"/"snv+sv"/"sv+snv" for "snp+sv") which
# _normalize_variants_choice folds back into one of these three tokens. Keeping
# the canonical set tight means downstream branches stay easy to reason about.
_AOU_VARIANT_ALIASES: dict[str, str] = {
    "sv": "sv",
    "snp": "snp",
    "snv": "snp",
    "snp+sv": "snp+sv",
    "snv+sv": "snp+sv",
    "sv+snp": "snp+sv",
    "sv+snv": "snp+sv",
}


def _normalize_variants_choice(variants: str) -> str:
    """Map a user-facing --variants token to one of the three canonical choices.

    Accepts the technical-term variation "snv" wherever "snp" is meaningful,
    and both orderings of the +-separated joint form. Raises if the token is
    none of those — argparse already filters at the CLI boundary but the
    Python entrypoint is also reachable from tests and notebooks.
    """
    canonical = _AOU_VARIANT_ALIASES.get(variants)
    if canonical is None:
        raise ValueError(
            f"variants must be one of {sorted(_AOU_VARIANT_ALIASES)}; got {variants!r}"
        )
    return canonical


def _aou_metadata_equivalent(existing: dict[str, object], current: dict[str, object]) -> bool:
    """Compare run-metadata dicts with legacy defaults for source/split fields.

    - Older runs predate the "variants" key; assume the historical SV default.
    - Older runs predate the "test_fraction" key; assume 0 (no holdout).
    - Older runs predate the "marginal_screen_min_abs_z" key; assume 0 (no screen),
      which is also the global ModelConfig default.

    Solver and posterior-diagnostic fields intentionally have no
    compatibility defaults: a missing value means the fit was produced under
    materially different settings and should be rerun.
    """
    if existing == current:
        return True
    existing = dict(existing)
    current = dict(current)
    if "variants" not in existing:
        existing["variants"] = "sv"
    if "test_fraction" not in existing:
        existing["test_fraction"] = 0.0
    if "marginal_screen_min_abs_z" not in existing:
        existing["marginal_screen_min_abs_z"] = 0.0
    return existing == current


def _split_merged_sample_table(
    merged_path: Path,
    *,
    test_fraction: float,
    random_seed: int,
) -> tuple[Path, Path | None]:
    """Split the merged sample table into train + held-out test rows.

    Splitting is deterministic in `(random_seed, sample_id)`: a SHA-256 hash
    maps each row's sample_id into [0, 1); rows below `test_fraction` go to
    the test partition, everyone else to train. This way reruns with the same
    seed produce the same split, and adding/removing rows only moves the few
    affected rows (not the whole assignment).

    Returns (train_path, test_path). test_path is None when test_fraction is
    zero, in which case the original file is used unchanged.
    """
    if not (0.0 <= test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in [0, 1); got {test_fraction!r}")
    if test_fraction == 0.0:
        return merged_path, None

    import hashlib

    base_name = merged_path.name
    train_path = merged_path.with_name(base_name.replace(".tsv", ".train.tsv", 1))
    test_path = merged_path.with_name(base_name.replace(".tsv", ".test.tsv", 1))
    salt = f"sv-pgs/split/seed={random_seed}".encode()

    train_lines: list[str] = []
    test_lines: list[str] = []
    with merged_path.open("rt", encoding="utf-8") as handle:
        header = handle.readline()
        train_lines.append(header)
        test_lines.append(header)
        for raw_line in handle:
            if not raw_line.strip():
                continue
            # sample_id is always column 0 (merge_pcs_into_sample_table emits
            # it as the leading column); slicing rather than full csv parse
            # keeps this O(bytes-in-row) for the 100k-row biobank case.
            sample_id = raw_line.split("\t", 1)[0]
            digest = hashlib.sha256(salt + b"|" + sample_id.encode()).digest()
            bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
            (test_lines if bucket < test_fraction else train_lines).append(raw_line)

    train_path.write_text("".join(train_lines), encoding="utf-8")
    test_path.write_text("".join(test_lines), encoding="utf-8")
    log(
        f"  train/test split (seed={random_seed}, test_fraction={test_fraction}): "
        f"{len(train_lines) - 1:,} train rows, {len(test_lines) - 1:,} test rows"
    )
    return train_path, test_path

DEFAULT_COVARIATES = [
    "age_at_observation_start",
    "age_squared",
    "gender_concept_id",
    "race_concept_id",
    "ethnicity_concept_id",
]

# Categorical OMOP fields that are one-hot expanded during phenotype
# preparation (see all_of_us._add_one_hot_omop_categorical_covariates).
# The raw column is popped from the sample table and replaced by one
# `<name>_<concept_id>` column per observed level. The covariate list
# must match the actual on-disk columns, so we expand each raw name back
# into the set of one-hot columns by reading the sample table header.
_OMOP_ONE_HOT_PREFIXES = (
    "gender_concept_id",
    "race_concept_id",
    "ethnicity_concept_id",
)


def _expand_one_hot_covariates(
    covariates: list[str], sample_table_path: Path
) -> list[str]:
    with sample_table_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    header_set = set(header)
    expanded: list[str] = []
    for name in covariates:
        if name in _OMOP_ONE_HOT_PREFIXES and name not in header_set:
            matches = [
                col for col in header
                if col.startswith(name + "_") and col[len(name) + 1:].isdigit()
            ]
            if not matches:
                raise ValueError(
                    f"covariate {name!r} is missing from {sample_table_path} "
                    "and no one-hot expansion columns were found"
                )
            expanded.extend(matches)
        else:
            expanded.append(name)
    return expanded


AOU_MAX_OUTER_ITERATIONS = 20
AOU_TEST_FRACTION = 0.2
AOU_MARGINAL_SCREEN_MIN_ABS_Z = 1.5
# Keep scheduled beta-variance refreshes outside the default AoU fit window;
# final posterior diagnostics are disabled below for the same memory budget.
AOU_BETA_VARIANCE_UPDATE_INTERVAL = AOU_MAX_OUTER_ITERATIONS + 1
AOU_FINAL_POSTERIOR_DIAGNOSTICS = False
AOU_VALIDATE_FIRST_ITERATION = True
# TODO: tune at runtime based on cohort size / GPU budget. Use a small
# non-zero default so the sample-space preconditioner is active and the
# first validation iteration produces useful early signal.
AOU_SAMPLE_SPACE_PRECONDITIONER_RANK = 32
# Per-epoch monitoring cadence (holdout metrics only; not used for model
# selection). Default 1 so training_history.tsv gets a row every epoch.
AOU_VALIDATION_INTERVAL = 1


@dataclass(slots=True)
class _PlinkCacheWarmup:
    executor: ThreadPoolExecutor
    future: Future[tuple[Any, Path | None]]
    original_compute: Any


def _start_plink_cache_warmup(
    *,
    sources: list[tuple[str, Path]],
    config: ModelConfig,
    sample_table_path: Path,
    covariates: list[str],
) -> _PlinkCacheWarmup | None:
    """Start the exact AoU PLINK stats/int8 cache pass the loader will need.

    The unified loader processes sources in order. For SNP+SV runs that means
    cached VCFs are opened first and PLINK stats/int8 construction begins only
    after the VCF work is finished. This warmup computes the same PLINK sample
    index vector up front, starts that cache pass in the background, and
    temporarily makes the loader wait on that future instead of launching a
    second cache writer for the same key.
    """
    if not any(kind == "vcf" for kind, _ in sources):
        return None
    plink_sources = [path for kind, path in sources if kind == "plink1"]
    if len(plink_sources) != 1:
        return None

    import sv_pgs.io as _io
    from sv_pgs.genotype import PlinkRawGenotypeMatrix
    from sv_pgs.io import VariantStatistics

    source_sample_ids: list[list[str]] = []
    plink_metadata = None
    for kind, path in sources:
        if kind == "vcf":
            ids = _io._read_vcf_sample_ids(path)
        else:
            plink_metadata = _io._load_plink1_metadata(path)
            ids = list(plink_metadata.sample_ids)
        source_sample_ids.append(ids)
    if plink_metadata is None:
        return None

    common_set = set(source_sample_ids[0])
    for ids in source_sample_ids[1:]:
        common_set &= set(ids)
    seen_common: set[str] = set()
    common_ordered: list[str] = []
    for sample_id in source_sample_ids[0]:
        if sample_id in common_set and sample_id not in seen_common:
            seen_common.add(sample_id)
            common_ordered.append(sample_id)
    if not common_ordered:
        raise RuntimeError("No samples in common across genotype sources.")

    sample_table_spec = _io._inspect_delimited_table(sample_table_path)
    resolved_sample_id_column = _io._resolve_sample_id_column(
        table_spec=sample_table_spec,
        requested_sample_id_column="auto",
        available_sample_ids=common_ordered,
    )
    sample_table, _, _ = _io._build_sample_table(
        table_spec=sample_table_spec,
        sample_id_column=resolved_sample_id_column,
        target_column="target",
        covariate_columns=covariates,
        available_sample_ids=common_ordered,
    )
    plink_path = plink_sources[0]
    plink_source_index = next(index for index, (kind, _) in enumerate(sources) if kind == "plink1")
    plink_sample_indices = np.asarray(
        _io._align_sample_ids(
            expected_sample_ids=sample_table.sample_ids,
            available_sample_ids=source_sample_ids[plink_source_index],
            context=f"plink1 source {plink_path.name}",
        ),
        dtype=np.intp,
    )
    original_compute = _io.compute_plink_variant_statistics_cached

    def _compute_plink_cache() -> tuple[VariantStatistics, Path | None]:
        raw = PlinkRawGenotypeMatrix(
            bed_path=plink_path,
            sample_indices=plink_sample_indices,
            variant_count=plink_metadata.variant_count,
            total_sample_count=len(plink_metadata.sample_ids),
        )
        log(
            "  PLINK stats/int8 cache warmup started in background "
            + f"({plink_sample_indices.shape[0]:,} samples x {plink_metadata.variant_count:,} variants)"
        )
        return original_compute(
            raw,
            bed_path=plink_path,
            sample_indices=plink_sample_indices,
            config=config,
        )

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aou-plink-cache")
    try:
        future = executor.submit(_compute_plink_cache)
    except BaseException:
        # If submission itself fails (e.g. interpreter shutting down), shut the
        # pool down rather than leaking the worker thread.
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    warmup_config = config

    def _compute_or_wait(
        raw_genotypes: PlinkRawGenotypeMatrix,
        bed_path: Path,
        sample_indices: NDArray,
        config: ModelConfig,
    ) -> tuple[VariantStatistics, Path | None]:
        if (
            Path(bed_path).resolve() == plink_path.resolve()
            and float(config.minimum_scale) == float(warmup_config.minimum_scale)
            and np.array_equal(np.asarray(sample_indices, dtype=np.intp), plink_sample_indices)
        ):
            log("  unified loader reached PLINK source; waiting for background stats/int8 cache warmup")
            return future.result()
        return original_compute(
            raw_genotypes,
            bed_path=bed_path,
            sample_indices=sample_indices,
            config=config,
        )

    _io.compute_plink_variant_statistics_cached = _compute_or_wait
    return _PlinkCacheWarmup(
        executor=executor,
        future=future,
        original_compute=original_compute,
    )


def _finish_plink_cache_warmup(warmup: _PlinkCacheWarmup | None) -> None:
    if warmup is None:
        return
    import sv_pgs.io as _io

    _io.compute_plink_variant_statistics_cached = warmup.original_compute
    warmup.executor.shutdown(wait=True, cancel_futures=True)


# Process-level dedupe so `_log_cached_test_evals(...)` emits each cached
# work_dir at most once per Python process. Without this, a sweep over 19
# diseases would re-print every sibling's cached numbers 20 times (once in
# the upfront scan, once per per-disease entry that re-scans the parent).
_LOGGED_CACHED_EVALS: set[str] = set()


def _format_metric(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _emit_loud_test_banner(metrics: dict[str, object], *, label: str, source: str) -> None:
    """One-line `>>> TEST ... <<<` banner, matching pipeline.py's end-of-fit format.

    Trait family is inferred from which metric keys are present: test_auc =>
    binary, test_r2 => quantitative. Both prefixes coexist with the same
    `grep '>>> TEST'` recipe operators already use on live runs.
    """
    n = metrics.get("test_n", metrics.get("test_sample_count"))
    if "test_auc" in metrics or "test_log_loss" in metrics or "test_accuracy" in metrics:
        log(
            f"  >>> TEST [{label} | {source}] AUC={_format_metric(metrics.get('test_auc'))}  "
            f"log_loss={_format_metric(metrics.get('test_log_loss'))}  "
            f"accuracy={_format_metric(metrics.get('test_accuracy'))}  "
            f"n={_format_metric(n)} <<<"
        )
    elif "test_r2" in metrics or "test_rmse" in metrics or "test_mse" in metrics:
        log(
            f"  >>> TEST [{label} | {source}] R2={_format_metric(metrics.get('test_r2'))}  "
            f"RMSE={_format_metric(metrics.get('test_rmse'))}  "
            f"MSE={_format_metric(metrics.get('test_mse'))}  "
            f"n={_format_metric(n)} <<<"
        )


def _log_cached_test_evals_from_summary(work_dir: Path, *, label: str) -> bool:
    """Log every test_* metric from a cached summary.json.gz, if one exists."""
    summary_path = work_dir / "summary.json.gz"
    if not summary_path.exists():
        return False
    from sv_pgs.io import _open_text_file

    try:
        with _open_text_file(summary_path, "rt") as handle:
            payload = json.loads(handle.read())
    except (OSError, ValueError) as exc:
        log(f"  CACHED RESULT [{label}]: failed to read {summary_path.name}: {exc}")
        return False
    if not isinstance(payload, dict):
        log(f"  CACHED RESULT [{label}]: {summary_path.name} is not a JSON object")
        return False
    test_items = sorted(
        (key, value) for key, value in payload.items() if key.startswith("test_")
    )
    log(f"=== CACHED RESULT TEST EVALS [{label}] from {summary_path.name} ===")
    if not test_items:
        log("  (summary has no test_* metrics — fit likely ran without a held-out test set)")
        return True
    for key, value in test_items:
        log(f"  {key} = {_format_metric(value)}")
    _emit_loud_test_banner(dict(test_items), label=label, source="cached summary")
    return True


def _log_cached_test_evals_from_history(work_dir: Path, *, label: str) -> bool:
    """If only a partial CACHED FIT exists, surface the latest per-epoch test row.

    `training_history.tsv` is appended row-by-row during EM (one row per
    validation epoch) with columns documented in
    `_build_per_epoch_history_writer`. Even mid-fit / interrupted runs leave
    a usable last row, so reporting it at job start lets the operator see
    how far the previous run got without rereading the log.
    """
    history_path = work_dir / "training_history.tsv"
    if not history_path.exists():
        return False
    try:
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            import csv as _csv

            reader = _csv.DictReader(handle, delimiter="\t")
            last_row: dict[str, str] | None = None
            for row in reader:
                last_row = row
    except OSError as exc:
        log(f"  CACHED FIT [{label}]: failed to read {history_path.name}: {exc}")
        return False
    if last_row is None:
        return False

    def _coerce(raw: str | None) -> float | int | str | None:
        if raw is None or raw == "":
            return None
        try:
            ivalue = int(raw)
            if str(ivalue) == raw:
                return ivalue
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            return raw

    parsed: dict[str, object] = {key: _coerce(value) for key, value in last_row.items()}
    log(
        f"=== CACHED FIT TEST EVALS [{label}] from {history_path.name} "
        f"(epoch {parsed.get('epoch')}, no summary.json.gz yet) ==="
    )
    test_items = sorted(
        (key, value)
        for key, value in parsed.items()
        if key.startswith("test_") and value is not None
    )
    if not test_items:
        log("  (history has no test_* values yet — first validation epoch not reached)")
        return True
    for key, value in test_items:
        log(f"  {key} = {_format_metric(value)}")
    _emit_loud_test_banner(dict(test_items), label=label, source=f"history epoch {parsed.get('epoch')}")
    return True


def _log_cached_test_evals(work_dir: Path, *, label: str | None = None) -> bool:
    """Surface any cached test set numbers for `work_dir` at the very start of a job.

    Preference order:
      1) summary.json.gz (CACHED result, fit fully completed) — strongest signal.
      2) training_history.tsv (CACHED fit, possibly mid-EM) — best partial signal.

    Dedupes via `_LOGGED_CACHED_EVALS`: each resolved work_dir is logged at
    most once per Python process so single-disease + sweep entry points don't
    spam the same numbers repeatedly. Returns True iff something was logged
    *on this call* (so callers can decide whether to follow up with extra
    framing); a duplicate-suppressed call returns False.
    """
    try:
        key = str(work_dir.resolve())
    except OSError:
        key = str(work_dir)
    if key in _LOGGED_CACHED_EVALS:
        return False
    resolved_label = label if label is not None else work_dir.name
    logged = (
        _log_cached_test_evals_from_summary(work_dir, label=resolved_label)
        or _log_cached_test_evals_from_history(work_dir, label=resolved_label)
    )
    if logged:
        _LOGGED_CACHED_EVALS.add(key)
    return logged


def _has_loggable_test_evals(work_dir: Path) -> bool:
    """Cheap predicate: would `_log_cached_test_evals` actually print anything?

    A header-only `training_history.tsv` (created by a prior run that was
    interrupted before the first validation epoch) is technically present
    on disk but has no test values to surface; treating it as a hit would
    print empty `JOB START` banners. Same logic for a summary.json.gz that
    we can't parse — the inner call will report the read error itself, but
    we want the outer banner to be true.
    """
    summary_path = work_dir / "summary.json.gz"
    if summary_path.exists():
        return True
    history_path = work_dir / "training_history.tsv"
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8", newline="") as handle:
                import csv as _csv

                reader = _csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    # Any non-empty test_* cell makes this row worth surfacing.
                    if any(
                        key.startswith("test_") and value not in (None, "")
                        for key, value in row.items()
                    ):
                        return True
        except OSError:
            return False
    return False


def _log_all_cached_test_evals(base_dir: Path, *, header: str = "PRE-FLIGHT") -> int:
    """Scan base_dir for cached per-job results and dump every test eval upfront.

    Returns the number of cached jobs found (whether or not each one was
    actually re-emitted — duplicates are silently suppressed by
    `_log_cached_test_evals`). `header` is a short tag so the framing reads
    sensibly whether this is the sweep pre-flight or a single-disease entry.

    A candidate dir only counts if it actually has test eval content to
    surface; a stale header-only training_history.tsv from a prior
    interrupted run no longer trips an empty banner.
    """
    if not base_dir.exists():
        return 0
    candidates = sorted(
        candidate
        for candidate in base_dir.glob("*_results")
        if candidate.is_dir() and _has_loggable_test_evals(candidate)
    )
    if not candidates:
        return 0
    unlogged = [
        candidate
        for candidate in candidates
        if str(_safe_resolve(candidate)) not in _LOGGED_CACHED_EVALS
    ]
    if not unlogged:
        # Everything we'd print has already been emitted earlier in this
        # process — stay quiet rather than print an empty banner.
        return len(candidates)
    log(f"=== {header}: {len(unlogged)} CACHED JOB(S) FOUND IN {base_dir} ===")
    for candidate in unlogged:
        _log_cached_test_evals(candidate, label=candidate.name)
    log(f"=== {header}: end of cached-result summary ===")
    return len(candidates)


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def run_all_of_us(
    disease: str,
    chromosomes: list[int],
    output_base: str,
    variant_metadata_path: str | Path | None = None,
    n_pcs: int = 10,
    random_seed: int = 0,
    variants: str = "snp+sv",
) -> None:
    """Full AoU pipeline: download requested chromosomes, merge them, and run one fit.

    `variants` selects the genotype sources fed into the joint model:
      "snp+sv"  — joint (default): AoU srWGS SV VCFs + AoU microarray PLINK,
                  intersected to the 97k-sample SV cohort.
      "sv"      — AoU srWGS SV VCFs only (97k samples).
      "snp"     — AoU microarray PLINK only (~447k samples).

    "snv" is accepted as an alias for "snp" and the joint form may be
    written in either order ("sv+snp", "snv+sv", etc.); the metadata file
    always records the canonical form.

    `marginal_screen_min_abs_z` is a univariate marginal-|z| floor applied
    after the MAF filter and before the joint Bayesian fit. Variants with
    |z| below this threshold (residualized on covariates; null distribution
    ~ N(0,1)) are dropped. AoU runs use |z| >= 1.5 and 20 EM iterations for a
    high-quality point-estimate fit, while skipping posterior variance
    diagnostics so T4-sized GPUs avoid sample-space preconditioner OOMs.
    """
    max_outer_iterations = AOU_MAX_OUTER_ITERATIONS
    test_fraction = AOU_TEST_FRACTION
    marginal_screen_min_abs_z = AOU_MARGINAL_SCREEN_MIN_ABS_Z
    beta_variance_update_interval = AOU_BETA_VARIANCE_UPDATE_INTERVAL
    final_posterior_diagnostics = AOU_FINAL_POSTERIOR_DIAGNOSTICS
    validate_first_iteration = AOU_VALIDATE_FIRST_ITERATION
    sample_space_preconditioner_rank = AOU_SAMPLE_SPACE_PRECONDITIONER_RANK
    # Holdout monitoring cadence; users override via the module constant.
    # ModelConfig itself has no `aou_validation_interval` field so we fall back
    # to the AOU_VALIDATION_INTERVAL module default. Default produces a row
    # every epoch in training_history.tsv (monitoring only, not model selection).
    validation_interval = max(1, int(AOU_VALIDATION_INTERVAL))
    variants = _normalize_variants_choice(variants)
    chromosomes = _validate_aou_chromosomes(chromosomes)

    import os

    # Validate disease
    disease_def = resolve_disease_definition(disease)
    work_dir = Path(output_base)
    work_dir.mkdir(parents=True, exist_ok=True)

    from sv_pgs.progress import set_log_file, start_heartbeat
    log_path = work_dir / f"{disease_def.canonical_name}.{time.strftime('%Y%m%d_%H%M%S')}.log"
    set_log_file(log_path)

    # Preflight: validate disk, gcsfuse layout, JAX/CuPy memory policy, GPU.
    # Aborts loudly on any fatal_error so a misconfigured workbench fails fast
    # instead of half-staging into a broken cache. The shared cache dir lives
    # one level above per-disease work_dir, so that's what we preflight.
    _preflight_cache_dir = work_dir.parent / _LOCAL_CACHE_DIRNAME
    _preflight_report = check_aou_preflight(cache_dir=_preflight_cache_dir)
    log_preflight(_preflight_report)
    assert_preflight_ok(_preflight_report)

    # Dump any cached held-out test evals at the very start so a `tail -f`
    # on the log shows the already-known numbers before the (potentially
    # long) cache-validation / rerun work begins. We scan the parent base
    # dir so single-disease invocations (e.g. `./run.sh hypertension`)
    # still surface every *sibling* disease's cached numbers, not just
    # this one's. The dedupe set keeps the sweep entry point's earlier
    # upfront scan from being re-emitted here.
    base_for_scan = work_dir.parent if work_dir.parent != work_dir else work_dir
    found = _log_all_cached_test_evals(base_for_scan, header=f"JOB START [{disease_def.canonical_name}]")
    if found == 0:
        # Nothing in parent — fall back to just this disease's work_dir on
        # the off-chance it exists and parent didn't match the *_results
        # glob (e.g. an output dir whose name doesn't end in _results).
        _log_cached_test_evals(work_dir, label=disease_def.canonical_name)

    # Background sampler: emits a periodic main-thread stack + CPU/GPU/mem
    # snapshot so a stall in the main thread is no longer silent.
    start_heartbeat(60.0)

    from sv_pgs.genotype import require_gpu
    require_gpu()

    log(
        "=== ALL OF US PIPELINE ===  "
        + f"disease={disease_def.canonical_name}  chromosomes={chromosomes}  n_pcs={n_pcs}  cpus={os.cpu_count()}"
    )
    log(f"  SNOMED root: {disease_def.snomed_code} ({disease_def.snomed_concept_name})")
    log(f"  output: {work_dir}")
    fit_checkpoint_path = work_dir / "fit_checkpoint.npz"
    if fit_checkpoint_path.exists():
        log(f"  warm-start checkpoint found: {fit_checkpoint_path} (will resume if compatible)")
    config = ModelConfig(
        max_outer_iterations=max_outer_iterations,
        random_seed=random_seed,
        marginal_screen_min_abs_z=marginal_screen_min_abs_z,
        beta_variance_update_interval=beta_variance_update_interval,
        final_posterior_diagnostics=final_posterior_diagnostics,
        validation_interval=validation_interval,
        validate_first_iteration=validate_first_iteration,
        sample_space_preconditioner_rank=sample_space_preconditioner_rank,
        # AoU outer iterations dominate runtime; the default 10-minute solver
        # budget was tripping TR-Newton mid-iter on the 695k-variant problem,
        # forcing a PG-IRLS restart and wasting ~10 min per outer iter. Give
        # TR-Newton an hour to actually finish iter 1.
        solver_wall_clock_budget_s=3600.0,
        # Phase 4 LD-block / N-GPU rewrite: AoU runs route the sample-space
        # matvec through per-LD-block per-GPU dispatch via
        # :class:`sv_pgs.gpu_scheduler.GPUScheduler`. Other entry points keep
        # the legacy single-monolithic matmul path (default ``False``) until
        # the new path has been end-to-end verified there too.
        use_ld_blocks=True,
    )
    log(
        "  AoU fit policy: "
        + f"max_iter={max_outer_iterations}  test_fraction={test_fraction:g}  "
        + f"screen=|z|>={marginal_screen_min_abs_z:g}  beta_var_interval={beta_variance_update_interval}  "
        + f"final_diagnostics={final_posterior_diagnostics}  "
        + f"sample_space_preconditioner_rank={sample_space_preconditioner_rank}"
    )

    # Status summary: what's done vs what's left
    log("=== STATUS CHECK ===")
    try:
        from sv_pgs.progress import log_autotune_banner

        log_autotune_banner()
    except (ImportError, RuntimeError) as _autotune_banner_error:
        log(f"  auto-tune banner unavailable: {_autotune_banner_error}")
    sample_table_path = work_dir / f"{disease_def.canonical_name}.samples.tsv"
    merged_path = work_dir / f"{disease_def.canonical_name}.samples.with_pcs.tsv"
    _sample_metadata_path = sample_table_path.with_suffix(sample_table_path.suffix + ".metadata.json")
    log(
        "  phenotype table: "
        + ("DONE" if (sample_table_path.exists() and _sample_metadata_path.exists()) else "NEEDED")
    )
    log(f"  PC-merged table: {'DONE' if merged_path.exists() else 'NEEDED'}")
    ancestry_local = local_ancestry_predictions_path(work_dir)
    resolved_variant_metadata_path = Path(variant_metadata_path) if variant_metadata_path is not None else None
    log(f"  ancestry file:   {'DONE' if ancestry_local.exists() else 'NEEDED'}")
    if resolved_variant_metadata_path is None:
        log("  variant metadata: not supplied")
    else:
        log(f"  variant metadata: {resolved_variant_metadata_path}")

    # Migrate old caches: previous versions stored caches in work_dir/.sv_pgs_cache/
    # with keys computed from the OLD VCF path.  The current code looks next to the
    # VCF files with keys computed from the NEW path.  We symlink old files under
    # the NEW key names so existing caches are found without re-parsing.
    from sv_pgs.io import _is_vcf_cache_bundle_complete, _vcf_cache_dir, _vcf_cache_key, _vcf_cache_paths
    import pickle as _pickle
    old_cache_dir = work_dir / _LOCAL_CACHE_DIRNAME
    if old_cache_dir.exists():
        first_vcf = local_sv_vcf_path(chromosomes[0], work_dir)
        if first_vcf.exists():
            new_cache_dir = _vcf_cache_dir(first_vcf)
            if old_cache_dir.resolve() != new_cache_dir.resolve():
                new_cache_dir.mkdir(parents=True, exist_ok=True)
                # Build map: chromosome → old key by reading variants.pkl
                old_chr_to_key: dict[str, str] = {}
                for geno_file in old_cache_dir.glob("*.genotypes.npy"):
                    old_k = geno_file.name.removesuffix(".genotypes.npy")
                    var_pkl = old_cache_dir / f"{old_k}.variants.pkl"
                    if not var_pkl.exists():
                        continue
                    try:
                        with open(var_pkl, "rb") as fh:
                            cached_variant_records = _pickle.load(fh)
                        if cached_variant_records:
                            chr_val = str(getattr(cached_variant_records[0], "chromosome", "")).replace("chr", "")
                            old_chr_to_key[chr_val] = old_k
                    except (OSError, _pickle.UnpicklingError, ValueError, EOFError, AttributeError):
                        continue
                # Symlink old files under NEW key names
                migrated_chrs = 0
                for chrom in chromosomes:
                    vcf_path = local_sv_vcf_path(chrom, work_dir)
                    if not vcf_path.exists():
                        continue
                    new_key = _vcf_cache_key(vcf_path, config)
                    old_key_for_chrom = old_chr_to_key.get(str(chrom))
                    if old_key_for_chrom is None:
                        continue
                    # Check if new key already has a complete cache
                    new_geno = new_cache_dir / f"{new_key}.genotypes.npy"
                    if new_geno.exists():
                        continue
                    for suffix in (".genotypes.npy", ".variants.pkl", ".stats.npy", ".stats.npz", ".manifest.json"):
                        src = old_cache_dir / f"{old_key_for_chrom}{suffix}"
                        dst = new_cache_dir / f"{new_key}{suffix}"
                        if src.exists() and not dst.exists():
                            try:
                                dst.symlink_to(src.resolve())
                            except OSError:
                                pass
                    migrated_chrs += 1
                # Clean up .tmp_parallel directories from failed parallel parses
                for tmp_dir in new_cache_dir.glob("*.tmp_parallel"):
                    try:
                        shutil.rmtree(tmp_dir)
                    except OSError:
                        pass
                if migrated_chrs > 0:
                    log(f"  migrated {migrated_chrs} chromosome caches (old key → new key) from {old_cache_dir}")

    cached_chrs = []
    uncached_chrs = []
    for chrom in chromosomes:
        vcf_path = local_sv_vcf_path(chrom, work_dir)
        if not vcf_path.exists():
            uncached_chrs.append(f"chr{chrom}(no VCF)")
            continue
        cache_dir = _vcf_cache_dir(vcf_path)
        cache_paths = _vcf_cache_paths(vcf_path, config)
        has_cache = _is_vcf_cache_bundle_complete(cache_paths)
        has_partial = (cache_dir / f"{cache_paths.key}.inc.progress.json").exists() if cache_dir.exists() else False
        has_tmp = (cache_dir / f"{cache_paths.key}.tmp_parallel").exists() if cache_dir.exists() else False
        if has_cache:
            cached_chrs.append(f"chr{chrom}")
        elif has_partial:
            uncached_chrs.append(f"chr{chrom}(partial)")
        elif has_tmp:
            uncached_chrs.append(f"chr{chrom}(parallel-partial)")
        else:
            uncached_chrs.append(f"chr{chrom}")
    # Skip SV-VCF status lines when running --variants snp, since we won't
    # download or precache them at all.
    if variants in ("sv", "snp+sv"):
        log(f"  SV VCF cached ({len(cached_chrs)}): {', '.join(cached_chrs) if cached_chrs else 'none'}")
        log(f"  SV VCF needed ({len(uncached_chrs)}): {', '.join(uncached_chrs) if uncached_chrs else 'none — all cached!'}")
    if variants in ("snp", "snp+sv"):
        # Microarray PLINK is one trio; either all three files are local or
        # we re-download. Report the aggregate state once instead of three
        # confusing lines.
        array_bed = local_array_plink_path(work_dir)
        array_present = all(
            array_bed.with_suffix(f".{extension}").exists()
            for extension in ("bed", "bim", "fam")
        )
        log(f"  microarray PLINK trio: {'DONE' if array_present else 'NEEDED'}")
    log(f"  variants source: {variants}")
    summary_path = work_dir / "summary.json.gz"
    log(f"  model fitted:    {'DONE' if summary_path.exists() else 'NEEDED'}")
    log("===================")

    # Step 1: Prepare phenotype
    log("=== STEP 1: Prepare phenotype ===")
    sample_table_path = work_dir / f"{disease_def.canonical_name}.samples.tsv"
    # The writer in prepare_all_of_us_disease_sample_table emits the TSV
    # first and the .sql / .metadata.json sidecars second. An interrupted
    # run therefore leaves a partial TSV without sidecars on disk. Treat
    # the metadata sidecar (written last) as the completion marker so
    # truncated TSVs are rewritten instead of silently reused.
    sample_metadata_path = sample_table_path.with_suffix(
        sample_table_path.suffix + ".metadata.json"
    )
    if not sample_table_path.exists() or not sample_metadata_path.exists():
        if sample_table_path.exists() and not sample_metadata_path.exists():
            log(
                f"  existing sample table missing metadata sidecar; rewriting: {sample_table_path}"
            )
            try:
                sample_table_path.unlink()
            except OSError:
                pass
        prepare_all_of_us_disease_sample_table(
            request=AllOfUsDiseaseRequest(disease=disease),
            output_path=sample_table_path,
        )
    else:
        log(f"  sample table already exists: {sample_table_path}")

    # Step 2: Download and merge PCs
    log("=== STEP 2: Merge genomic PCs ===")
    ancestry_path = download_ancestry_preds(work_dir)
    merged_path = work_dir / f"{disease_def.canonical_name}.samples.with_pcs.tsv"
    merged_path, pc_cols = merge_pcs_into_sample_table(
        sample_table_path=sample_table_path,
        ancestry_path=ancestry_path,
        output_path=merged_path,
        n_pcs=n_pcs,
    )

    # Build covariate list. The phenotype preparation step one-hot encodes
    # gender/race/ethnicity concept_ids into `<name>_<id>` columns and drops
    # the raw column, so expand those entries against the merged table's
    # header before passing the list to the loader.
    covariates = _expand_one_hot_covariates(
        DEFAULT_COVARIATES + pc_cols, merged_path
    )
    log(f"  covariates ({len(covariates)}): {covariates}")

    summary_path = work_dir / "summary.json.gz"
    run_metadata_path = _aou_run_metadata_path(work_dir)
    run_metadata = _build_aou_run_metadata(
        disease=disease_def.canonical_name,
        chromosomes=chromosomes,
        n_pcs=n_pcs,
        pc_cols=pc_cols,
        covariates=covariates,
        max_outer_iterations=max_outer_iterations,
        random_seed=random_seed,
        variant_metadata_path=str(resolved_variant_metadata_path) if resolved_variant_metadata_path is not None else None,
        variants=variants,
        beta_variance_update_interval=beta_variance_update_interval,
        final_posterior_diagnostics=final_posterior_diagnostics,
        validation_interval=validation_interval,
        validate_first_iteration=validate_first_iteration,
        sample_space_preconditioner_rank=sample_space_preconditioner_rank,
        test_fraction=test_fraction,
        marginal_screen_min_abs_z=marginal_screen_min_abs_z,
    )
    if summary_path.exists():
        if run_metadata_path.exists():
            existing_run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
            if _aou_metadata_equivalent(existing_run_metadata, run_metadata):
                log(f"  unified fit already exists with matching configuration: {summary_path}")
                return
            log("  existing unified fit configuration differs; rerunning and overwriting outputs")
        else:
            log("  unified fit exists without run metadata; rerunning and overwriting outputs")

    log(f"=== STEP 3: Download genotype sources (variants={variants}) ===")
    # `sources` is the ordered list of (kind, local_path) tuples handed to the
    # multi-source loader. SV VCFs come first (matching the historical sample
    # order); the microarray PLINK trio, if requested, is appended as the
    # single PLINK source.
    sources: list[tuple[str, Path]] = []
    vcf_paths: list[Path] = []
    dataset = None
    test_dataset = None
    plink_cache_warmup: _PlinkCacheWarmup | None = None
    try:
        if variants in ("sv", "snp+sv"):
            for chrom in chromosomes:
                local_vcf = download_sv_vcf(chrom, work_dir)
                vcf_paths.append(local_vcf)
                sources.append(("vcf", local_vcf))
        if variants in ("snp", "snp+sv"):
            log("  downloading microarray PLINK trio...")
            array_bed = download_array_plink(work_dir)
            sources.append(("plink1", array_bed))

        # Surface where each genotype source actually lives — gcsfuse paths
        # read at network speed (~36 MB/s) whereas local NVMe hits ~3 GB/s,
        # and historical runs have silently regressed by leaving a symlink
        # pointing into the gcsfuse mount. One log line per source makes
        # this immediately visible at run start.
        for kind, src_path in sources:
            try:
                resolved = Path(src_path).resolve()
            except OSError:
                resolved = Path(src_path)
            try:
                where = "gcsfuse" if is_gcsfuse_path(resolved) else "local-hot"
            except Exception:  # noqa: BLE001
                where = "unknown"
            log(f"  source storage: {kind:<7s} {src_path} -> {resolved} [{where}]")

        # Carve a deterministic held-out test partition off the merged sample
        # table before we load genotypes. The split uses sample_id+seed so
        # reruns are reproducible — and so two sibling runs (e.g. SV-only vs
        # SV+SNP) keep the same train/test assignment, which makes their
        # held-out AUCs comparable.
        train_table_path, test_table_path = _split_merged_sample_table(
            merged_path=merged_path,
            test_fraction=test_fraction,
            random_seed=random_seed,
        )

        if variants == "snp+sv":
            log("=== STEP 3.25: Start PLINK cache warmup beside VCF work ===")
            plink_cache_warmup = _start_plink_cache_warmup(
                sources=sources,
                config=config,
                sample_table_path=train_table_path,
                covariates=covariates,
            )
            if plink_cache_warmup is None:
                log("  PLINK cache warmup not applicable")

        if vcf_paths:
            log("=== STEP 3.5: Parallel VCF precache ===")
            from sv_pgs.io import precache_vcfs_parallel
            try:
                precache_vcfs_parallel(vcf_paths, config)
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError("parallel VCF precache failed") from exc
        else:
            log("=== STEP 3.5: skipping VCF precache (no VCF sources) ===")

        log("=== STEP 4: Load unified genome-wide TRAIN dataset ===")
        try:
            dataset = load_multi_source_dataset_from_files(
                sources=[(kind, str(path)) for kind, path in sources],
                config=config,
                sample_table_path=str(train_table_path),
                sample_id_column="auto",
                target_column="target",
                covariate_columns=covariates,
                variant_metadata_path=resolved_variant_metadata_path,
            )
        finally:
            _finish_plink_cache_warmup(plink_cache_warmup)
            plink_cache_warmup = None
        if test_table_path is not None:
            log("=== STEP 4b: Load held-out TEST dataset ===")
            # Same sources, same variant order, different sample subset. The
            # multi-source loader's intersection logic will further restrict
            # to samples present in every source — for SV-only that's a no-op,
            # for SV+microarray that's the SV ∩ array intersection.
            # Pass train's variant_stats + variant_records straight through so
            # the test load skips its own ~10-110 min PLINK variant-stats
            # streaming pass entirely. The downstream pipeline only consumes
            # test_dataset.{genotypes, covariates, targets, sample_ids}; stats
            # and records are kept consistent with train so any consumer that
            # reads them sees one model, one standardization.
            test_dataset = load_multi_source_dataset_from_files(
                sources=[(kind, str(path)) for kind, path in sources],
                config=config,
                sample_table_path=str(test_table_path),
                sample_id_column="auto",
                target_column="target",
                covariate_columns=covariates,
                variant_metadata_path=resolved_variant_metadata_path,
                precomputed_variant_stats=dataset.variant_stats,
                precomputed_variant_records=dataset.variant_records,
            )
        inferred_trait = TraitType.BINARY if len(np.unique(dataset.targets)) <= 2 else TraitType.QUANTITATIVE
        config.trait_type = inferred_trait

        log("=== STEP 5: Fit unified genome-wide model ===")
        log(f"  freeing dataset overhead before fit...  mem={mem()}")
        # Release the large variant_records list from the dataset — run_training_pipeline
        # will re-create training records from variant_stats inside fit()
        gc.collect()
        log(f"  memory after gc: {mem()}")
        register_fit_checkpoint_path(config, fit_checkpoint_path)
        run_training_pipeline(
            dataset=dataset,
            config=config,
            output_dir=work_dir,
            test_dataset=test_dataset,
        )
        # Atomic publish so a crash mid-write (or a concurrent disease
        # sweep racing on the same path) never leaves a truncated metadata
        # JSON that downstream tools would mis-parse as the run record.
        _run_metadata_tmp = run_metadata_path.with_name(
            f"{run_metadata_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        try:
            _run_metadata_tmp.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
            os.replace(_run_metadata_tmp, run_metadata_path)
        except BaseException:
            try:
                _run_metadata_tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    finally:
        _finish_plink_cache_warmup(plink_cache_warmup)
        if dataset is not None:
            del dataset
        if test_dataset is not None:
            del test_dataset
        # Keep VCFs on disk for cache / reruns (do NOT delete)
        release_process_memory()
        log("=== UNIFIED FIT CLEANUP DONE ===")

    log("=== ALL OF US PIPELINE COMPLETE ===")


def _detect_gpu_count() -> int:
    """Best-effort GPU count via CuPy runtime.

    Returns 0 when CuPy is missing, the CUDA driver/runtime isn't loadable,
    or the host has no GPUs. Single-disease subprocesses still enforce CUDA
    at pipeline entry and fail loudly instead of fitting on CPU.
    """
    from sv_pgs.genotype import _detect_cuda_device_count

    return _detect_cuda_device_count()


def _build_disease_subprocess_cmd(
    *,
    disease: str,
    chromosomes: list[int],
    disease_output_dir: Path,
    variant_metadata_path: str | Path | None,
    n_pcs: int,
    random_seed: int,
    variants: str,
) -> list[str]:
    """Construct the argv for a single-disease subprocess.

    Each child reinvokes the same Python interpreter via `python -m sv_pgs`
    with the existing single-disease `run-all-of-us` flags, so the child's
    code path is unchanged from a normal single-disease run.
    """
    import sys as _sys

    # Normalize chromosomes to the canonical comma-separated form the CLI
    # accepts (it also takes "1-22" / "5" but the comma form is the
    # widest-coverage choice and lossless for arbitrary subsets).
    chromosomes_arg = ",".join(str(c) for c in chromosomes)
    cmd = [
        _sys.executable,
        "-m",
        "sv_pgs",
        "run-all-of-us",
        "--disease",
        disease,
        "--chromosomes",
        chromosomes_arg,
        "--output-dir",
        str(disease_output_dir),
        "--n-pcs",
        str(int(n_pcs)),
        "--random-seed",
        str(int(random_seed)),
        "--variants",
        variants,
    ]
    if variant_metadata_path is not None:
        cmd.extend(["--variant-metadata", str(variant_metadata_path)])
    return cmd


def _run_disease_subprocess(
    *,
    disease: str,
    cmd: list[str],
    gpu_id: int | None,
) -> tuple[str, int, str]:
    """Spawn one disease subprocess and wait for it.

    Returns (disease, returncode, stderr_tail). When `gpu_id` is None the
    child inherits the parent's GPU environment unchanged, so one fit can use
    every visible CUDA device through the sharded genotype cache.
    """
    env = dict(os.environ)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log(
        f"  [scheduler] spawning disease={disease} pid_parent={os.getpid()} "
        f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<inherit>')}"
    )
    # Capture stderr so we can include the tail in the per-disease failure
    # log; let stdout stream through to the parent's stdout so `tail -f`
    # still surfaces the child's STEP banners + per-epoch test rows.
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=None,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr_tail = ""
    try:
        _, stderr_out = proc.communicate()
        if stderr_out:
            # Echo full stderr through so the parent log preserves it, but
            # keep a short tail for the structured failure summary.
            for line in stderr_out.splitlines():
                log(f"  [{disease} stderr] {line}")
            tail_lines = stderr_out.splitlines()[-20:]
            stderr_tail = "\n".join(tail_lines)
    except BaseException:
        # Parent received a signal (e.g. Ctrl-C); SIGTERM the child, give it
        # a moment to shut down cleanly, then SIGKILL stragglers. This keeps
        # GPU memory from being held by a zombie when the user aborts a sweep.
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        except OSError:
            pass
        raise
    return disease, int(proc.returncode), stderr_tail


def run_all_of_us_all_diseases(
    chromosomes: list[int],
    output_base: str,
    variant_metadata_path: str | Path | None = None,
    n_pcs: int = 10,
    random_seed: int = 0,
    variants: str = "snp+sv",
    max_parallel_gpus: int | None = None,
) -> int:
    """Run the AoU pipeline for every disease in DISEASE_DEFINITIONS.

    Each disease writes into its own subdirectory `<canonical_name>_results/`
    under `output_base`. By default the sweep runs one disease at a time and
    each fit sees all visible GPUs, so the model-level multi-GPU sharded
    genotype operator accelerates the fit itself.

    Pool sizing:
      - `max_parallel_gpus=None` (default) runs sequentially with all visible
        GPUs available to each fit.
      - `max_parallel_gpus=1` also runs sequentially.
      - `max_parallel_gpus>1` explicitly trades per-fit multi-GPU for
        per-disease concurrency by pinning each subprocess to one GPU.

    Returns the process exit code: 0 on full success, 1 if any disease
    subprocess failed (the sweep still finishes the rest before returning).
    """
    base_dir = Path(output_base)
    base_dir.mkdir(parents=True, exist_ok=True)
    # Print every already-cached test set eval up front so operators see the
    # known numbers immediately even if the sweep is about to spend hours on
    # remaining diseases or cache re-validation.
    _log_all_cached_test_evals(base_dir, header="SWEEP PRE-FLIGHT")

    diseases = list(DISEASE_DEFINITIONS)
    n_diseases = len(diseases)

    detected_gpus = _detect_gpu_count()
    if max_parallel_gpus is None:
        pool_size = 1
        if detected_gpus > 0:
            scheduling_source = f"auto-detect ({detected_gpus} GPU(s); each fit uses all visible GPUs)"
        else:
            scheduling_source = "auto-detect (no GPUs visible; sequential)"
    elif detected_gpus <= 0:
        pool_size = 1
        scheduling_source = "user override ignored for no-GPU host; sequential"
    else:
        requested = max(1, int(max_parallel_gpus))
        pool_size = max(1, min(requested, detected_gpus, n_diseases))
        scheduling_source = f"user override (--max-parallel-gpus={max_parallel_gpus})"

    waves = (n_diseases + pool_size - 1) // pool_size
    log(
        f"multi-GPU scheduling: {scheduling_source}, sweeping {n_diseases} "
        f"diseases -> {pool_size} concurrent fit(s) ({waves} wave(s))"
    )

    # When pool_size==1, fall back to in-process sequential execution to
    # preserve the historical behavior (avoids the subprocess overhead and
    # keeps the existing logs/heartbeat in one process tree). Multi-GPU
    # parallelism only kicks in when pool_size > 1.
    failures: list[tuple[str, str]] = []
    if pool_size == 1:
        for disease_definition in diseases:
            disease_output_dir = base_dir / f"{disease_definition.canonical_name}_results"
            log(
                f"=== TOP-20 LOOP: starting {disease_definition.canonical_name} -> {disease_output_dir} ==="
            )
            try:
                run_all_of_us(
                    disease=disease_definition.canonical_name,
                    chromosomes=chromosomes,
                    output_base=str(disease_output_dir),
                    variant_metadata_path=variant_metadata_path,
                    n_pcs=n_pcs,
                    random_seed=random_seed,
                    variants=variants,
                )
            except (RuntimeError, ValueError, OSError) as exc:
                log(
                    f"=== TOP-20 LOOP: {disease_definition.canonical_name} FAILED: {exc} ==="
                )
                failures.append((disease_definition.canonical_name, str(exc)))
        if failures:
            details = "; ".join(f"{name}: {msg}" for name, msg in failures)
            log(f"run_all_of_us_all_diseases: {len(failures)} disease(s) failed: {details}")
            return 1
        return 0

    # Explicit subprocess fan-out. Process diseases in waves of `pool_size`,
    # round-robin assigning each disease to a GPU id in [0, pool_size). This is
    # only used when the caller explicitly asks to run multiple diseases at
    # once; the default path above keeps all GPUs available to each single fit.
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_completed

    pending = list(enumerate(diseases))
    with _TPE(max_workers=pool_size, thread_name_prefix="aou-disease-sched") as executor:
        future_to_disease: dict = {}
        for submission_index, disease_definition in pending:
            gpu_id = submission_index % pool_size if detected_gpus > 0 else None
            disease_output_dir = base_dir / f"{disease_definition.canonical_name}_results"
            cmd = _build_disease_subprocess_cmd(
                disease=disease_definition.canonical_name,
                chromosomes=chromosomes,
                disease_output_dir=disease_output_dir,
                variant_metadata_path=variant_metadata_path,
                n_pcs=n_pcs,
                random_seed=random_seed,
                variants=variants,
            )
            log(
                f"=== TOP-20 LOOP: queued {disease_definition.canonical_name} "
                f"CUDA_VISIBLE_DEVICES={gpu_id if gpu_id is not None else '<inherit>'} "
                f"-> {disease_output_dir} ==="
            )
            future = executor.submit(
                _run_disease_subprocess,
                disease=disease_definition.canonical_name,
                cmd=cmd,
                gpu_id=gpu_id,
            )
            future_to_disease[future] = disease_definition.canonical_name

        try:
            for future in _as_completed(future_to_disease):
                disease_name = future_to_disease[future]
                try:
                    name, returncode, stderr_tail = future.result()
                except (OSError, subprocess.SubprocessError) as exc:
                    log(f"=== TOP-20 LOOP: {disease_name} subprocess error: {exc} ===")
                    failures.append((disease_name, f"subprocess error: {exc}"))
                    continue
                if returncode != 0:
                    log(
                        f"=== TOP-20 LOOP: {name} FAILED (exit {returncode}); stderr tail:\n{stderr_tail} ==="
                    )
                    failures.append((name, f"exit {returncode}: {stderr_tail[-500:]}"))
                else:
                    log(f"=== TOP-20 LOOP: {name} completed successfully ===")
        except BaseException:
            # Cancel anything that hasn't started yet so a Ctrl-C doesn't
            # leave the executor draining a backlog of new spawns. Running
            # subprocesses are signalled inside _run_disease_subprocess.
            for future in future_to_disease:
                future.cancel()
            raise

    if failures:
        details = "; ".join(f"{name}: {msg}" for name, msg in failures)
        log(f"run_all_of_us_all_diseases: {len(failures)} disease(s) failed: {details}")
        return 1
    return 0
