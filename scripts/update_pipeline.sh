#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Check if "pipeline only" argument is provided
PIPELINE_ONLY=false
if [[ "${1:-}" == "pipeline-only" || "${1:-}" == "--pipeline-only" ]]; then
    PIPELINE_ONLY=true
    echo "Skipping environment updates, updating pipeline only."
fi

# Update the pipeline
git fetch origin
git checkout main
git reset --hard origin/main

# Skip environment updates if pipeline-only mode is enabled
if [[ "$PIPELINE_ONLY" == false ]]; then
    echo "Updating environment..."

    # Update micromamba
    ~/.local/bin/micromamba self-update

    # Build a fresh environment prefix and switch the `align` env symlink over to
    # it, instead of mutating the live env in place. An in-place update corrupts the
    # env whenever micromamba cannot remove a busy file; building fresh + swapping
    # never has that problem. See env/build_env.sh.
    bash ./env/build_env.sh

    # The fresh env has no pipeline package yet; register it (editable, --no-deps).
    # A pipeline-only update skips this: the existing editable install already
    # tracks this checkout, which the git reset above just updated.
    ~/.local/bin/micromamba run -n align pip install -e . --no-deps
fi
