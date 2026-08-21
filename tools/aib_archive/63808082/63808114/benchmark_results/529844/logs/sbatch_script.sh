#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=kimi-k3-mxfp4-gb300-disagg-1p1d-tp8dcp8pp2-tp8dcp8-pushpp-dspark-ret12288-c8-oci
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/common/gitlab_runner/misunp/results/2026-08-20T21_20_55-07_00_406083914/%j/logs/sweep_%j.log

#SBATCH --nodes=6

#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1



#SBATCH --segment=6


#SBATCH --account=coreai_comparch_inferencex
#SBATCH --time=04:00:00
#SBATCH --partition=batch


#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"model_loading","description":"Very large model will take extra loading time."}}'



#SBATCH --cpus-per-task=72



#SBATCH --mem=0



#SBATCH --gres=gpu:4



#SBATCH --qos=normal




# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Orchestrator runs on HOST (needs srun access), spawns containerized workers
# Config: /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/common/gitlab_runner/misunp/j-E7tcfZW/1/compute/anna/infrastructure/AIB/sweep_workdir_406083914/patched_srt_config.yaml
# Generated: 20260821_042202

set -e

# Setup directories
SRTCTL_SOURCE="/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/common/gitlab_runner/misunp/j-E7tcfZW/1/compute/anna/infrastructure/AIB/sweep_workdir_406083914/srt_slurm_repo"
OUTPUT_BASE="/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/common/gitlab_runner/misunp/results/2026-08-20T21_20_55-07_00_406083914"
OUTPUT_DIR="${OUTPUT_BASE}/${SLURM_JOB_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Export for Python orchestrator to use the same paths
export SRTCTL_OUTPUT_DIR="${OUTPUT_DIR}"

# Redirect stderr to stdout (SLURM captures both via --output)
exec 2>&1

echo "=========================================="
echo "🚀 srtctl Sweep Orchestrator"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Config: /scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/common/gitlab_runner/misunp/j-E7tcfZW/1/compute/anna/infrastructure/AIB/sweep_workdir_406083914/patched_srt_config.yaml"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "CPUs per node: ${SLURM_JOB_CPUS_PER_NODE:-unknown}"
echo "CPUs on orchestrator node: ${SLURM_CPUS_ON_NODE:-unknown}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE:-unknown}"
echo "Container: vllm/vllm-openai:nightly-5a4c8d99242e9e069b604d0e9b969e77f7dd501d"
echo "Start: $(date)"
echo "=========================================="
echo ""

# submit.py always copies the original submission config to OUTPUT_DIR/config.yaml.
# For override jobs it also writes the resolved runtime config below.
echo "Runtime config: ${OUTPUT_DIR}/config.yaml"

# Get head node
HEAD_NODE=$(scontrol show hostnames ${SLURM_NODELIST} | head -n1)
echo "Head node: ${HEAD_NODE}"

# Set source directory for container mounts (/configs)
export SRTCTL_SOURCE_DIR="${SRTCTL_SOURCE}"



echo ""
echo "Preparing srtctl environment..."

# Use compute-arch uv from srt-slurm/bin (installed by make setup ARCH=<compute_arch>)
# This avoids arch mismatch when login node (x86_64) != compute node (aarch64)
export PATH="${SRTCTL_SOURCE}/bin:$HOME/.local/bin:$PATH"

# Use a compute-specific venv so we don't pick up the login node's .venv/
# (which may be built for a different arch, e.g. x86_64 login vs aarch64 compute)
export UV_PROJECT_ENVIRONMENT="${SRTCTL_SOURCE}/.venv-compute"

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Run 'make setup ARCH=<compute_arch>' first."
    echo "  e.g., make setup ARCH=aarch64  (for GB200/Grace compute nodes)"
    exit 1
fi

echo "Using uv $(uv --version)..."

if [ -n "${DYNAMO_WHEEL_NAME:-}" ] || [ "${SRTCTL_PREFETCH_AI_DYNAMO:-0}" = "1" ]; then
    if [ -f "${SRTCTL_SOURCE}/src/srtctl/runtime_scripts/dynamo_wheels.py" ]; then
        uv run --no-project --python "${DYNAMO_PYTHON_VERSION:-3.12}" --with pip \
            python "${SRTCTL_SOURCE}/src/srtctl/runtime_scripts/dynamo_wheels.py" prefetch
    else
        echo "WARNING: ${SRTCTL_SOURCE}/src/srtctl/runtime_scripts/dynamo_wheels.py not found"
    fi
fi



echo "Running orchestrator on host (with srun access)..."
echo ""

# Run orchestrator on the HOST (not in container) so it has access to srun
# The orchestrator will spawn containerized workers
# Output goes to SLURM's --output (sweep_%j.log)
set +e  # Temporarily disable exit-on-error to capture exit code
cd "${SRTCTL_SOURCE}"
# Lock-protected venv sync to avoid concurrent .venv-compute teardown/recreate
# races across simultaneous SLURM jobs sharing this directory. uv may decide
# the existing venv is invalid (broken python interpreter symlink) and tear
# it down + rebuild; if N jobs do that at once they stomp each other with
# "Directory not empty" / "Failed to spawn 'python -m'" / missing CACHEDIR.TAG
# errors. The flock serializes only the venv init; the lock is released
# before the actual sweep so jobs still run in parallel.
flock -x "${SRTCTL_SOURCE}/.venv-compute.lock" uv sync --python 3.12 --no-dev
uv run --python 3.12 --no-dev --no-sync -m srtctl.cli.do_sweep "${OUTPUT_DIR}/config.yaml"
EXIT_CODE=$?
set -e  # Re-enable exit-on-error

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Sweep completed successfully"
else
    echo "✗ Sweep failed (exit code: $EXIT_CODE)"
fi
echo "End: $(date)"
echo "=========================================="

exit ${EXIT_CODE}