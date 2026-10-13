"""Sample-independent, ordered feature identity. Ambiguous legacy stores cannot transfer scores.

Alleles are compared exactly in their supplied representation. This deliberately rejects
alternative representations rather than guessing normalization or strand remapping.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from sv_pgs.content_identity import file_digest

IDENTITY_FILE = "FEATURE_IDENTITY.json"
FORMAT = "svpgs-feature-identity-v1"


def chromosome_identity(*, genome_build, chromosome, positions, refs, alts, identifiers,
                        ends, sv_types, sv_lengths, codes_per_unit, value_origin) -> dict:
    fields = (positions, refs, alts, identifiers, ends, sv_types, sv_lengths, codes_per_unit, value_origin)
    if not genome_build or len({len(f) for f in fields}) != 1:
        raise ValueError("Feature identities require a genome build and aligned fields")
    digest = hashlib.sha256()
    for pos, ref, alt, identifier, end, kind, length, units, origin in zip(*fields):
        if not ref or not alt or int(pos) < 1 or float(units) <= 0:
            raise ValueError("Invalid feature allele, position or units")
        if str(alt).startswith("<") and end is None and not np.isfinite(length):
            raise ValueError("Symbolic SV identity needs END or SVLEN")
        record = [str(genome_build), str(chromosome), int(pos), str(ref), str(alt), str(identifier),
                  None if end is None else int(end), kind, None if not np.isfinite(length) else float(length),
                  float(units), float(origin), "ALT_count", "exact_input_representation"]
        digest.update(json.dumps(record, separators=(",", ":"), allow_nan=False).encode() + b"\n")
    return {"format": FORMAT, "genome_build": genome_build, "chromosome": chromosome,
            "records": len(positions), "sha256": digest.hexdigest()}


def layout_digest(root, manifest) -> str:
    root = Path(root)
    identity = root / IDENTITY_FILE
    layout = {"format": FORMAT, "chromosomes": manifest["chromosomes"], "counts": manifest["record_counts"],
              "recalibrated": manifest.get("attributes", {}).get("recalibrated", False),
              "feature_identity": file_digest(identity) if identity.exists() else "legacy_unverified", "columns": {}}
    # Bind the actual stored decoder and row ordering as well as the source alleles.
    for chrom in manifest["chromosomes"]:
        for name in ("pos", "ref_len", "alt_len", "variant_class", "codes_per_unit", "value_origin", "group_first", "id_bytes", "id_offsets"):
            directory = root / "variants" / chrom / name
            for path in sorted(p for p in directory.rglob("*") if p.is_file()):
                layout["columns"][path.relative_to(root).as_posix()] = file_digest(path)
    return hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()


def portable(root, manifest) -> bool:
    path = Path(root) / IDENTITY_FILE
    if not path.exists():
        return False
    entries = json.loads(path.read_text())
    return (set(entries) == set(manifest["chromosomes"]) and all(
        entry.get("format") == FORMAT and entry.get("genome_build") not in (None, "", "unspecified")
        and entry.get("chromosome") == chrom and entry.get("records") == count
        for chrom, count in zip(manifest["chromosomes"], manifest["record_counts"])
        for entry in [entries[chrom]]))
