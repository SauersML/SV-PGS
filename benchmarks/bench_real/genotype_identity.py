"""Workspace-only exact genotype identity matching, with aggregate-only output.

Inputs are workspace NPZ files with string ``ids``, string ``variants`` (ordered
CHROM:POS:REF:ALT keys), and uint8 ``genotypes`` [people, sites]: 0/1/2, 3 missing.
Use at least 2048 common, high-quality autosomal SNPs, identical alleles/order in
both inputs. No participant identifiers or genotypes are ever printed. Outputs
remain on workspace disk. No outcomes or model scores enter identity assignment.

The two best concordances are exact over every reference participant, not an
approximate nearest-neighbour search. A match requires >=99% concordance, >=95%
called overlap, a >=5 percentage point runner-up gap and one-to-one assignment.
Reference duplicate IDs and insufficient inputs fail closed. This is a genetic
identity link between pseudonymous study identifiers, not an identity lookup.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def shown(n):
    return str(int(n)) if n >= 40 else "<40"


def pack(genotypes):
    """Two genotype bits and one valid-call bit per site, 32 sites per word."""
    g = np.asarray(genotypes, dtype=np.uint8)
    if g.ndim != 2 or np.any(g > 3):
        raise ValueError("Invalid genotype matrix")
    n, sites = g.shape
    words = (sites + 31) // 32
    code = np.zeros((n, words), dtype=np.uint64)
    valid = np.zeros_like(code)
    for offset in range(32):
        part = g[:, offset::32]
        width = part.shape[1]
        code[:, :width] |= part.astype(np.uint64) << np.uint64(2 * offset)
        valid[:, :width] |= (part != 3).astype(np.uint64) << np.uint64(2 * offset)
    return code, valid


def exact_top_two(query, reference, min_overlap):
    """Return internal indices and concordances; compile only inside workspace."""
    from numba import njit, prange, types
    from numba.extending import intrinsic
    from llvmlite import ir

    @intrinsic
    def popcount(typingctx, value):
        signature = types.uint64(types.uint64)
        def codegen(context, builder, sig, args):
            fn = builder.module.declare_intrinsic("llvm.ctpop", [ir.IntType(64)])
            return builder.call(fn, args)
        return signature, codegen

    @njit(parallel=True)
    def search(qcode, qvalid, rcode, rvalid, sites, minimum):
        best_ids = np.full((len(qcode), 2), -1, dtype=np.int64)
        best_values = np.full((len(qcode), 2), -1.0)
        overlaps = np.zeros((len(qcode), 2), dtype=np.int64)
        for i in prange(len(qcode)):
            for j in range(len(rcode)):
                missing, differences = 0, 0
                rejected = False
                for k in range(qcode.shape[1]):
                    common = qvalid[i, k] & rvalid[j, k]
                    width = min(32, sites - 32 * k)
                    missing += width - int(popcount(common))
                    changed = qcode[i, k] ^ rcode[j, k]
                    differences += int(popcount((changed | (changed >> np.uint64(1))) & common))
                    # Any remaining sites can only increase mismatches/missingness.
                    if sites - missing < minimum or (best_values[i, 1] >= 0 and
                            differences > (1.0 - best_values[i, 1]) * sites + 1e-9):
                        rejected = True
                        break
                if rejected:
                    continue
                overlap = sites - missing
                concordance = 1.0 - differences / overlap
                if concordance > best_values[i, 0]:
                    best_values[i, 1] = best_values[i, 0]
                    best_ids[i, 1] = best_ids[i, 0]
                    overlaps[i, 1] = overlaps[i, 0]
                    best_values[i, 0], best_ids[i, 0], overlaps[i, 0] = concordance, j, overlap
                elif concordance > best_values[i, 1]:
                    best_values[i, 1], best_ids[i, 1], overlaps[i, 1] = concordance, j, overlap
        return best_ids, best_values, overlaps

    qcode, qvalid = pack(query)
    rcode, rvalid = pack(reference)
    return search(qcode, qvalid, rcode, rvalid, query.shape[1], min_overlap)


def assignments(indices, concordance, threshold=0.99, gap=0.05):
    """Reject ties, insufficient runner-up evidence and all duplicate assignments."""
    eligible = ((indices[:, 0] >= 0) & (indices[:, 1] >= 0)
                & (concordance[:, 0] >= threshold)
                & (concordance[:, 0] - concordance[:, 1] >= gap))
    unique, counts = np.unique(indices[eligible, 0], return_counts=True)
    duplicated = unique[counts > 1]
    accepted = eligible & ~np.isin(indices[:, 0], duplicated)
    ambiguous = (concordance[:, 0] >= threshold) & ~accepted
    return accepted, ambiguous


def distribution(values):
    values = np.asarray(values)
    values = values[np.isfinite(values) & (values >= 0)]
    if len(values) < 40:
        return {"people": "<40", "distribution": "suppressed"}
    return {"people": shown(len(values)), "quantiles_01_10_50_90_99":
            np.quantile(values, [0.01, 0.1, 0.5, 0.9, 0.99]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("reference")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not Path("/home/jupyter").is_dir() or not Path("/home/jupyter/workspace/vwb-aou-datasets-controlled").is_dir():
        raise RuntimeError("Execution restricted to the AoU workspace")
    from numba import set_num_threads
    set_num_threads(args.threads)
    os.umask(0o077)
    with np.load(args.query, allow_pickle=False) as q, np.load(args.reference, allow_pickle=False) as r:
        qids, rids = q["ids"], r["ids"]
        qg, rg = q["genotypes"], r["genotypes"]
        if len(set(qids)) != len(qids) or len(set(rids)) != len(rids):
            raise ValueError("Duplicate input identifiers; no map written")
        if (qg.shape[1] < 2048 or qg.shape[1] != rg.shape[1] or
                not np.array_equal(q["variants"], r["variants"])):
            raise ValueError("Insufficient or discordant SNP keys; no map written")
        if len(qids) != len(qg) or len(rids) != len(rg) or min(len(qids), len(rids)) < 40:
            raise ValueError("Insufficient or inconsistent participant rows; no map written")
        indices, concordance, overlap = exact_top_two(qg, rg, int(np.ceil(0.95 * qg.shape[1])))
        accepted, ambiguous = assignments(indices, concordance)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(out.suffix + ".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["research_id", "sequencing_id"])
            for i in np.flatnonzero(accepted):
                writer.writerow([rids[indices[i, 0]], qids[i]])
        temporary.replace(out)
        # Internal QC rows must never be opened by the operator.
        np.savez_compressed(out.with_suffix(".private_qc.npz"), indices=indices,
                            concordance=concordance, overlap=overlap, accepted=accepted)
        summary = {"matched": shown(accepted.sum()), "ambiguous": shown(ambiguous.sum()),
                   "unmatched": shown((~accepted & ~ambiguous).sum()),
                   "best_concordance": distribution(concordance[:, 0]),
                   "second_best_concordance": distribution(concordance[:, 1])}
        out.with_suffix(".aggregate_qc.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Exception messages or tracebacks can contain data; emit only the class.
        print("Identity matching failed; details retained in workspace: " + type(error).__name__)
        raise SystemExit(1)
