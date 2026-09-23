#!/bin/bash
#SBATCH -J pipeline
#SBATCH -o ./sbatch_output/pipeline-%j.out
#SBATCH -e ./sbatch_output/pipeline-%j.err
# Resource directives (-c/-t/--mem/--gres/--account/...) are passed on the
# sbatch command line by run_pipeline.sh, derived from the config's sbatch_init.

# Arguments go straight to the pipeline, which parses them, resolves its own run
# directory and moves the logs above into it. ALIGN_PYTHON is inherited
# from run_pipeline.sh; the default covers a direct sbatch of this script.
# ALIGN_INIT_JOB_ID marks this job as the pipeline's launcher, so the run
# directory is keyed on it; a bare SLURM_JOB_ID may come from an unrelated
# allocation (e.g. an interactive session) and is ignored.
export ALIGN_INIT_JOB_ID="$SLURM_JOB_ID"
${ALIGN_PYTHON:-$HOME/.local/bin/micromamba run -n align python3} -m align_pipeline.init_pipeline "$@"
