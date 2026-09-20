#!/bin/bash
# bench-tox: extract the analysis lines (restricted list) from the public 1kGP 30x phased SNV/INDEL/SV panel,
# every record, no thresholds, as one plink bed and one GCTA-format GRM per chromosome.
# Chromosomes are claimed by atomic mkdir, so tasks on several runners share the work; a claim is released on failure.
set -uo pipefail
RESTRICTED=/projects/standard/hsiehph/sauer354/svpgs-team/restricted/tox_dream
OUT=/scratch.global/sauer354/svpgs-team/bench-tox/geno
PANEL=/scratch.global/sauer354/svpgs-team/public/kgp_phased_panel_20220422
IDS=$RESTRICTED/derived/analysis_lines.txt
PARALLEL=${1:?number of chromosomes to run at once}
DEADLINE=${2:?unix time by which a chromosome must be finished}
mkdir -p "$OUT" && chmod 700 "$OUT" /scratch.global/sauer354/svpgs-team/bench-tox
module load plink/2.0.0-a7.1 >/dev/null 2>&1
awk 'BEGIN{print "#IID"} {print}' "$IDS" > "$OUT/keep.txt" && chmod 600 "$OUT/keep.txt"

one() {
    local chrom=$1 claim="$OUT/claim_chr$1" remaining
    [ -f "$OUT/chr$chrom.done" ] && return 0
    if ! mkdir "$claim" 2>/dev/null; then
        # A claim whose owner's deadline has passed was left by a killed task: take it over.
        [ -f "$claim/deadline" ] && [ "$(cat "$claim/deadline")" -lt "$(date +%s)" ] || return 0
        echo "$DEADLINE" > "$claim/deadline"
    fi
    echo "$DEADLINE" > "$claim/deadline"
    remaining=$(( DEADLINE - $(date +%s) ))
    if [ "$remaining" -le 0 ]; then rm -f -- "$claim/deadline"; rmdir "$claim"; return 0; fi
    # Work on scratch, not TMPDIR: on acl42 TMPDIR is RAM and counts against the task's memory cap.
    local tmp=$OUT/work_chr$chrom
    rm -rf -- "${tmp:?}"; mkdir -p "$tmp"
    if timeout --foreground "$remaining" plink2 --threads 1 --memory 3000 \
            --vcf "$PANEL/1kGP_high_coverage_Illumina.chr$chrom.filtered.SNV_INDEL_SV_phased_panel.vcf.gz" \
            --keep "$OUT/keep.txt" --set-all-var-ids '@:#:$r:$a' --new-id-max-allele-len 23 truncate \
            --make-bed --out "$tmp/chr$chrom" > "$tmp/plink.log" 2>&1 \
       && timeout --foreground "$(( DEADLINE - $(date +%s) ))" plink2 --threads 1 --memory 3000 --bfile "$tmp/chr$chrom" \
            --make-grm-bin --out "$tmp/grm_chr$chrom" >> "$tmp/plink.log" 2>&1; then
        mv "$tmp"/chr$chrom.* "$tmp"/grm_chr$chrom.* "$OUT/" && cp "$tmp/plink.log" "$OUT/chr$chrom.plink.log"
        chmod 600 "$OUT"/chr$chrom.* "$OUT"/grm_chr$chrom.*
        touch "$OUT/chr$chrom.done"
    else
        cp "$tmp/plink.log" "$OUT/chr$chrom.failed.log" 2>/dev/null
    fi
    rm -rf -- "${tmp:?}"
    rm -f -- "$claim/deadline"; rmdir "$claim" 2>/dev/null
}

export -f one
export OUT PANEL DEADLINE
printf '%s\n' 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 | xargs -P "$PARALLEL" -I{} bash -c 'one {}'
ls "$OUT"/*.done 2>/dev/null | wc -l
