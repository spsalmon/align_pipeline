#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# just in case, reset the repository to the latest version of main
git fetch origin
git checkout main
git reset --hard origin/main

# Update micromamba
~/.local/bin/micromamba self-update

# Build the environment into a fresh prefix and switch the `align` env symlink
# over to it. See env/build_env.sh for why we never mutate the live env.
bash ./env/build_env.sh

# Register the pipeline package (and the align command) in the new
# env. Editable, so it tracks this checkout; --no-deps because the locked env
# already holds every dependency and pip must not touch the pinned set.
~/.local/bin/micromamba run -n align pip install -e . --no-deps

mkdir -p ./sbatch_output
