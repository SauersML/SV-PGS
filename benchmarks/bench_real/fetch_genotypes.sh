#!/bin/bash
# Download the public 1kGP 3,202-sample phased SNV/INDEL/SV panel (NYGC high coverage, 2022-04-22) and the HGSVC2
# PanGenie genotypes, check the panel files against the EBI manifest md5, and write MAGE-sample subsets per autosome:
#   <root>/data/geno/chr<c>.bcf           panel sites polymorphic in MAGE
#   <root>/data/geno/pangenie_chr<c>.bcf  PanGenie alleles of at least 50 bp polymorphic in MAGE
# Usage: fetch_genotypes.sh <benchmark root> <shared public-data dir> <threads>
set -euo pipefail
ROOT=$1
PUBLIC=$2
THREADS=$3
PANEL=$PUBLIC/kgp_phased_panel_20220422
PANEL_URL=http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV
PANGENIE_URL=http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC2/release/v2.0/PanGenie_results
PANGENIE=$ROOT/data/hgsvc2/20201217_pangenie_merged_bi_nosnvs.vcf.gz
mkdir -p "$PANEL" "$ROOT/data/geno" "$ROOT/data/hgsvc2" "$ROOT/data/kgp"
curl -s -o "$PANEL/20220804_manifest.txt" "$PANEL_URL/20220804_manifest.txt"
curl -s -o "$ROOT/data/kgp/20130606_g1k_3202_samples_ped_population.txt" \
  http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt
tail -n +2 "$ROOT/data/mage/sample_library_info/sample.metadata.MAGE.v1.0.txt" | cut -f4 | sort -u > "$ROOT/data/geno/mage_samples.txt"
for file in "$(basename "$PANGENIE")" "$(basename "$PANGENIE").tbi"; do
  [ -s "$ROOT/data/hgsvc2/$file" ] || curl -s --retry 5 -o "$ROOT/data/hgsvc2/$file" "$PANGENIE_URL/$file"
done
for chrom in $(seq 1 22); do
  name=1kGP_high_coverage_Illumina.chr${chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz
  for file in "$name" "$name.tbi"; do
    expected=$(awk -v f="$file" '$1==f{print $2}' "$PANEL/20220804_manifest.txt")
    if [ ! -s "$PANEL/$file" ] || [ "$(md5sum "$PANEL/$file" | cut -d' ' -f1)" != "$expected" ]; then
      curl -s --retry 5 -o "$PANEL/$file" "$PANEL_URL/$file"
    fi
    [ "$(md5sum "$PANEL/$file" | cut -d' ' -f1)" == "$expected" ] || { echo "md5 mismatch $file"; exit 1; }
  done
  bcftools view --threads "$THREADS" -S "$ROOT/data/geno/mage_samples.txt" -c 1:minor -Ob -o "$ROOT/data/geno/chr${chrom}.bcf" "$PANEL/$name"
  bcftools index -f "$ROOT/data/geno/chr${chrom}.bcf"
  # 50 bp is the structural-variant size definition, see STRUCTURAL_VARIANT_MIN_BP in build_dataset.py.
  bcftools view --threads "$THREADS" -r "chr$chrom" -S "$ROOT/data/geno/mage_samples.txt" -c 1:minor \
    -i 'strlen(REF)>=50 || strlen(ALT)>=50' -Ob -o "$ROOT/data/geno/pangenie_chr${chrom}.bcf" "$PANGENIE"
  bcftools index -f "$ROOT/data/geno/pangenie_chr${chrom}.bcf"
done
md5sum "$PANEL"/*.vcf.gz > "$PANEL/md5_checked.txt"
