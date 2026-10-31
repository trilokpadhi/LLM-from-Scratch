# Shell environment for running this repo on GSU ARCTIC.
# Source it from ~/.bashrc (see arctic/README.md). Jobs submitted through
# slurm_launch.py also source it, via ENV_SETUP_COMMANDS.

# uv is installed per-user (there is no uv module on ARCTIC).
export PATH="$HOME/.local/bin:$PATH"

# Big files live on /scratch; the repo and its .venv stay in $HOME.
# utils.py derives DATA_DIR=<root>/data and MODEL_DIR=<root>/ckpts from this.
export DL_ALCHEMY_SCRATCH_ROOT="/scratch/users/$USER/dl_alchemy"
export UV_CACHE_DIR="/scratch/users/$USER/uv_cache"
# The uv cache and .venv are on different filesystems, so copy instead of hardlink.
export UV_LINK_MODE=copy
