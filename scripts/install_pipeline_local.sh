#!/bin/bash
set -euo pipefail

# Local install (no micromamba; any OS with bash -- Git Bash/WSL on Windows):
# create the base conda env, then install the pipeline + deps editable from
# pyproject. `conda run` avoids needing `conda activate` inside a script.
cd "$(dirname "$0")/.."

conda env create -f env/environment_local.yml
conda run -n align_pipeline_local pip install -e ".[dev]"

echo ">> Done. Activate with: conda activate align_pipeline_local"
