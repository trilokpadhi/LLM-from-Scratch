"""Short 1-GPU L40S run that checks the whole ARCTIC path end to end.

Run from the repo root:  uv run python -m arctic.smoke_train
"""

from slurm_launch import launch_job
from train import TrainConfig, train


EXPERIMENT_KEY = "arctic-smoke"


def main():
    launch_job(
        "train(TrainConfig(num_train_sequences=20_000, "
        "run_name_suffix=EXPERIMENT_KEY, wandb_tags=(EXPERIMENT_KEY,)))",
        queue="arctic",
        gpus=1,
        mem=32,
        cpus=8,
    )


if __name__ == "__main__":
    main()
