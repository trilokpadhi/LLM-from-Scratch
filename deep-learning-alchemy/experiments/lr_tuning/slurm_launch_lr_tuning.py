"""Slurm version of launch_lr_tuning.py: one L40S per learning rate on ARCTIC.

Run from the repo root:  uv run python -m experiments.lr_tuning.slurm_launch_lr_tuning
Plot afterwards with:    uv run python -m experiments.lr_tuning.plot_lr_tuning
"""

from slurm_launch import launch_job_array
from train import TrainConfig, train

from experiments.lr_tuning.launch_lr_tuning import EXPERIMENT_KEY, LEARNING_RATES


# The six L40S GPUs are shared, so cap how many this sweep holds at once.
MAX_CONCURRENT = 3


def function_calls():
    return [
        f"train(TrainConfig(learning_rate={learning_rate!r}, "
        "run_name_suffix=EXPERIMENT_KEY, wandb_tags=(EXPERIMENT_KEY,)))"
        for learning_rate in LEARNING_RATES
    ]


def main():
    launch_job_array(
        function_calls(),
        queue="arctic",
        gpus=1,
        mem=32,
        cpus=8,
        max_concurrent=MAX_CONCURRENT,
    )


if __name__ == "__main__":
    main()
