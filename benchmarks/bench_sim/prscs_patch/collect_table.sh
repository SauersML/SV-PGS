#!/bin/bash
# collect_table.sh: the per-(scenario, method) rows of paired2.py into results/baselines_vs_svpgs.tsv
R=/scratch.global/sauer354/svpgs-team/agents/baselines-genome/results
out=$R/baselines_vs_svpgs.tsv
printf "scenario\tmethod\treference\tr2\tslope\tref_r2\td_r2\td_r2_lo\td_r2_hi\tgenetic_r2\tref_genetic_r2\td_genetic_r2\td_genetic_r2_lo\td_genetic_r2_hi\toracle_r2\n" > $out.tmp
cat $R/paired_table/*.tsv 2>/dev/null | sort -k1,1 -k2,2 >> $out.tmp
mv $out.tmp $out
echo "$(($(wc -l < $out) - 1)) rows in $out"
