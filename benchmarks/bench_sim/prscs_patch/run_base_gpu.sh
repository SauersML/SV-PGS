#!/bin/bash
# run_base_gpu.sh <task script>: an LDpred2 or SBayesRC bench-sim task with its LD products on the GPU (BASELINES_GPU=1)
# and the cohort genotypes prefetched into page cache by 16 streams; temporary files stay on scratch (TMPDIR below)
A=/scratch.global/sauer354/svpgs-team/agents/baselines-genome
V=/projects/standard/hsiehph/sauer354/svpgs-team/venv-gpu
C=/scratch.global/sauer354/svpgs-team/bench-sim/v7/cohort/chr22
export LD_LIBRARY_PATH="$(ls -d $V/lib/python3.12/site-packages/nvidia/*/lib | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export BASELINES_GPU=1 CUDA_PATH=$V/lib/python3.12/site-packages/nvidia/cuda_runtime CUPY_CACHE_DIR=$A/cupy_cache
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
F=$C/observed_truth.npy
size=$(stat -c %s $F); seg=$(( size / 16 / 16777216 + 1 )); t0=$(date +%s)
for i in $(seq 0 15); do dd if=$F of=/dev/null bs=16M skip=$((i*seg)) count=$seg status=none & done; wait
echo "prefetched $F ($size bytes) in $(( $(date +%s) - t0 )) s"
mkdir -p $A/tmp/job_$SLURM_JOB_ID
sed "s#/projects/standard/hsiehph/sauer354/svpgs-team/venv-cpu#$V#" $A/slurm/$1 > $A/tmp/job_$SLURM_JOB_ID/task.sh
TMPDIR=$A/tmp/job_$SLURM_JOB_ID bash $A/tmp/job_$SLURM_JOB_ID/task.sh
