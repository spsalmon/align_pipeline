#!/bin/bash
#SBATCH -J postprocess_autofocus
#SBATCH -o ../sbatch_output/postprocess_autofocus.out
#SBATCH -e ../sbatch_output/postprocess_autofocus.err
#SBATCH -c 64
#SBATCH -t 12:00:00
#SBATCH --mem=64G

INPUT_DIR="/mnt/towbin.data/shared/spradhan/20250929_embryogenesis_316_0uM_500uM_P0_20x_Orca_III/raw/"
OUTPUT_DIR="/mnt/towbin.data/shared/spradhan/20250929_embryogenesis_316_0uM_500uM_P0_20x_Orca_III/raw_best_plane/"
CHANNEL=0

~/.local/bin/micromamba run -n align python3 postprocess_autofocus.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --channel "$CHANNEL"
