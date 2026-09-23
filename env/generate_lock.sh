#!/bin/bash
set -euo pipefail

# Regenerate the lock files from environment.yml.
#
# Two files are produced and BOTH are committed:
#   - conda-lock.yml        canonical conda-lock solve (render input, source of truth)
#   - conda-linux-64.lock   explicit render -> the file install/update actually consume
#
# install_pipeline.sh and update_pipeline.sh both read conda-linux-64.lock in two
# phases: micromamba installs the conda packages from it, and pip installs the pip
# packages from its `# pip <name> @ <url>#sha256=...` comment lines. Keeping conda
# and pip separate avoids combined-transaction failures when files are held open by
# running jobs or a Jupyter kernel.
#
# We deliberately do NOT render the `env` kind (conda-linux-64.lock.yml): it used to
# be depended on, then stopped being generated, which silently broke env updates.

cd "$(dirname "$0")"

# Drop stale artifacts so the directory only holds the current lock files.
rm -f conda-lock.yml conda-linux-64.lock conda-linux-64.lock.yml

# 1. Solve the environment -> conda-lock.yml
# Driven through conda_lock_patched.py rather than the conda-lock executable so
# that manylinux_2_27 wheels (nvidia-nccl-cu13, via xgboost) can be resolved.
~/.local/bin/micromamba run -n align_pipeline python conda_lock_patched.py lock -f environment.yml -p linux-64 --micromamba

# 2. Render the explicit single-platform lock -> conda-linux-64.lock
~/.local/bin/micromamba run -n align_pipeline conda-lock render -k explicit
