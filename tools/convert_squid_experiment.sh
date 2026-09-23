#!/bin/bash

#SBATCH -J convert_squid
#SBATCH -o ../sbatch_output/convert_squid.out
#SBATCH -e ../sbatch_output/convert_squid.err
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH --mem=16GB

INPUT_DIR="/mnt/towbin.data/shared/fdell/Starvation_survival_wild_strains/20260501_SQUID0_wBT499to506_starvation_survival/squid_raw/"
OUTPUT_DIR="/mnt/towbin.data/shared/fdell/Starvation_survival_wild_strains/20260501_SQUID0_wBT499to506_starvation_survival/raw/"

~/.local/bin/micromamba run -n align_pipeline python3 convert_squid_experiment.py --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR
