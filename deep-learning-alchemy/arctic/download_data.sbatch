#!/bin/bash
# Download the course's tokenized DCLM data (~19.7 GB) on a CPU node.
# Submit from the repo root:  mkdir -p slurmjobs && sbatch arctic/download_data.sbatch

#SBATCH --job-name=dl-alchemy-data
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=slurmjobs/download_data-%j.out

set -euo pipefail
source arctic/env.sh
uv run python -m download_data
