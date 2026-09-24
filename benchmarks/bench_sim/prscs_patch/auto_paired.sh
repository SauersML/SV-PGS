#!/bin/bash
# auto_paired.sh: every 3 min, for each scenario whose SV-PGS prediction (svpgs_v5, then v4, then v2) and all
# baseline arms' predictions exist, submit one paired2 job (paired2.py picks the same reference) and print its job id;
# a scenario is redone when a preferred reference lands. Refreshes the combined table each cycle.
done_list=/private/tmp/claude-501/-Users-user-SV-PGS/4fbcbd69-2c02-4adf-a5dc-c7745c5dd139/scratchpad/paired_v2_submitted.txt
touch $done_list
A=/scratch.global/sauer354/svpgs-team/agents/baselines-genome
T=/scratch.global/sauer354/svpgs-team/lead/bench-sim-truth/results/truth
M=ldpred2_auto_allvar,ldpred2_auto,sbayesrc,sbayesrc_annotfree,prscs_auto_allvar,prscs_auto
while true; do
  have=$(/Users/user/msi-node/msi -n acl42 "ls $T/svpgs_v5/*/prediction.npz $T/svpgs_v4/*/prediction.npz $T/svpgs_v2/*/prediction.npz 2>/dev/null; for m in ldpred2_auto_allvar ldpred2_auto sbayesrc prscs_auto_allvar prscs_auto; do ls $A/results/\$m/*/prediction.npz 2>/dev/null; done; $A/collect_table.sh >/dev/null" 2>/dev/null)
  [ -z "$have" ] && { sleep 180; continue; }
  for s in 000 001 004 005 008 009 013 015 018; do
    if echo "$have" | grep -q "svpgs_v5/scenario_$s/prediction.npz"; then ref=v5
    elif echo "$have" | grep -q "svpgs_v4/scenario_$s/prediction.npz"; then ref=v4
    elif echo "$have" | grep -q "svpgs_v2/scenario_$s/prediction.npz"; then ref=v2
    else continue; fi
    grep -qx "$s:$ref" $done_list && continue
    [ $ref = v2 ] && grep -qx "$s" $done_list && continue  # done against v2 by the earlier watcher
    n=$(echo "$have" | grep "/agents/baselines-genome/results/" | grep -c "/scenario_$s/prediction.npz")
    [ "$n" -ge 5 ] || continue
    id=$(/Users/user/msi-node/msi -n acl42 "$A/bg_paired2.sh ${ref}s$s $s $M" 2>/dev/null | grep -E "^[0-9]+$")
    [ -n "$id" ] && { echo "$s:$ref" >> $done_list; echo "paired2 vs svpgs_$ref submitted for $s: job $id (log paired2_${ref}s$s.log)"; }
  done
  sleep 180
done
