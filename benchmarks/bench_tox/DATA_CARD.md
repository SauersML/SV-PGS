# bench-tox data card (metadata only; no per-line data in this repository)

## Phenotypes (restricted)
- **Location:** /projects/standard/hsiehph/sauer354/svpgs-team/restricted/tox_dream/, mode 700. PROVENANCE.md there lists the terms and rules.
- **Files and sha256:**

  | File | Synapse | sha256 |
  |---|---|---|
  | ToxChallenge_CytotoxicityData_Train_Subchal1_Extended.txt | syn2186574 | 33b889ac… |
  | ToxChallenge_Subchall1_Test_Data.txt | syn2280490 | 038855f6… |
  | ToxChallenge_Covariates.txt | syn1917748 | fa55d398… |

- **Size:** 884 lines × 106 compounds pooled; 793 analysis lines in the 1kGP 30× panel.
- **Sealing:** 34 confirmation compounds, list sha256 d154565847dc7915…
- **Cite:** Eduati et al. 2015, Nat Biotechnol 33:933; Abdo et al. 2015, Environ Health Perspect 123:458; Synapse syn1761567.

## Genotypes (public)
- **Panel:** 1kGP high-coverage phased SNV/INDEL/SV panel 20220422 (Byrska-Bishop et al. 2022), EBI, md5-checked.
- **Built files:** per-chromosome plink2 beds and GCTA GRMs for the 793 lines, every record, in /scratch.global/sauer354/svpgs-team/bench-tox/geno/ (mode 700). They're built by `build_genotypes.sh` in the lane's code directory.
- **Recorded later:** the splits' sha256 once they're built.
