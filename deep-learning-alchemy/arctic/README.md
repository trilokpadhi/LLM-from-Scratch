# Running on GSU ARCTIC

Notes for running this repo on the ARCTIC Slurm cluster instead of Modal. The
six L40S GPUs are all on node `acidsgcn010`, in partitions `qGPU24` (1-day
limit), `qGPU48`, and `qGPU120`. Each training run uses one GPU.

Layout:

- repo and `.venv`: `~/LLM-from-Scratch/deep-learning-alchemy`
- data (~19.7 GB) and checkpoints: `/scratch/users/$USER/dl_alchemy`
- uv cache: `/scratch/users/$USER/uv_cache`

These paths come from [env.sh](env.sh).

## One-time setup

```bash
# uv (installs to ~/.local/bin; ARCTIC has no uv module)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone -b deep-learning-alchemy https://github.com/trilokpadhi/LLM-from-Scratch.git ~/LLM-from-Scratch
cd ~/LLM-from-Scratch/deep-learning-alchemy
echo 'source ~/LLM-from-Scratch/deep-learning-alchemy/arctic/env.sh' >> ~/.bashrc
source arctic/env.sh

uv python install 3.10
uv sync
uv run wandb login          # paste the API key from https://wandb.ai/authorize

mkdir -p slurmjobs
sbatch arctic/download_data.sbatch   # ~19.7 GB onto /scratch; log in slurmjobs/
```

## Running

Always launch from the repo root, on a login node:

```bash
uv run python -m arctic.smoke_train                          # quick 1-GPU check
uv run python -m experiments.lr_tuning.slurm_launch_lr_tuning # LR sweep, 3 GPUs at a time
uv run python -m experiments.lr_tuning.plot_lr_tuning         # plot from W&B
```

Monitor with `squeue -u $USER`; job logs go to `slurmjobs/`, and each run
prints its `WANDB_RUN_URL`. For new experiments, copy
[slurm_launch_lr_tuning.py](../experiments/lr_tuning/slurm_launch_lr_tuning.py):
build one `train(TrainConfig(...))` string per run and pass them to
`launch_job_array(..., queue="arctic", gpus=1)`.

The `arctic` queue is defined in `QUEUE_CONFIGS` in
[slurm_launch.py](../slurm_launch.py). Change its `partition` and `time`
together if a run needs more than a day.
