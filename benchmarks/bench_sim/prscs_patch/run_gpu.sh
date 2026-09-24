#!/bin/bash
# run_gpu2.sh <task script>: a bench-sim task run whole under the GPU venv (PRSCS_DEVICE=gpu: the submission's LD
# products, PRS-CS's symmetrization and its MCMC block updates on the device), temporary files on the job's local disk
A=/scratch.global/sauer354/svpgs-team/agents/baselines-genome
V=/projects/standard/hsiehph/sauer354/svpgs-team/venv-gpu
C=/scratch.global/sauer354/svpgs-team/bench-sim/v7/cohort/chr22
export LD_LIBRARY_PATH="$(ls -d $V/lib/python3.12/site-packages/nvidia/*/lib | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PRSCS_DEVICE=gpu PRSCS_GIG=vector
export CUDA_PATH=$V/lib/python3.12/site-packages/nvidia/cuda_runtime CUPY_CACHE_DIR=$A/cupy_cache
export TMPDIR=${TMPDIR:-/tmp}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# the cohort's genotype file read into the page cache by 16 parallel streams (a single stream from scratch runs at
# ~100 MB/s; the harness then reads it from memory)
F=$C/observed_truth.npy
size=$(stat -c %s $F); seg=$(( size / 16 / 16777216 + 1 )); t0=$(date +%s)
for i in $(seq 0 15); do dd if=$F of=/dev/null bs=16M skip=$((i*seg)) count=$seg status=none & done; wait
echo "prefetched $F ($size bytes) in $(( $(date +%s) - t0 )) s"
sed "s#/projects/standard/hsiehph/sauer354/svpgs-team/venv-cpu#$V#" $A/slurm/$1 > $TMPDIR/task.sh
bash $TMPDIR/task.sh
