#!/bin/bash
# validate_multi.sh: PRS-CS's sampler, 5 iterations from one seed, on the CPU, one GPU and two GPUs; compares them
A=/scratch.global/sauer354/svpgs-team/agents/baselines-genome
V=/projects/standard/hsiehph/sauer354/svpgs-team/venv-gpu
export LD_LIBRARY_PATH="$(ls -d $V/lib/python3.12/site-packages/nvidia/*/lib | paste -sd:)"
export CUDA_PATH=$V/lib/python3.12/site-packages/nvidia/cuda_runtime CUPY_CACHE_DIR=$A/cupy_cache
export PYTHONPATH=$A/pylib OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TMPDIR=${TMPDIR:-/tmp}
nvidia-smi --query-gpu=name --format=csv,noheader
unset PRSCS_DEVICE; $V/bin/python -u $A/validate_multi.py cpu $TMPDIR/cpu.npz
PRSCS_DEVICE=gpu CUDA_VISIBLE_DEVICES=0 $V/bin/python -u $A/validate_multi.py gpu $TMPDIR/gpu1.npz
PRSCS_DEVICE=gpu CUDA_VISIBLE_DEVICES=0,1 $V/bin/python -u $A/validate_multi.py gpu $TMPDIR/gpu2.npz
$V/bin/python - <<EOF
import numpy as np
c = np.load("$TMPDIR/cpu.npz")
for name in ("gpu1", "gpu2"):
    g = np.load(f"$TMPDIR/{name}.npz")
    rel = {k: float(np.abs(g[k] - c[k]).max() / np.abs(c[k]).max()) for k in ("beta", "psi", "sigma", "phi")}
    print(name, "vs cpu, max relative difference:", {k: f"{v:.1e}" for k, v in rel.items()}, "within 1e-10:", max(rel.values()) < 1e-10)
EOF
