#!/bin/bash
#SBATCH -J convert_nd2
#SBATCH -o ../sbatch_output/convert_nd2.out
#SBATCH -e ../sbatch_output/convert_nd2.err
#SBATCH -c 32
#SBATCH -t 48:00:00
#SBATCH --mem=64G

INPUT_DIR="/mnt/towbin.data/shared/spsalmon/20260807_134551_682_ZIVA_60x_col10_reporter/raw_nd2"
OUTPUT_DIR="/mnt/towbin.data/shared/spsalmon/20260807_134551_682_ZIVA_60x_col10_reporter/raw"
NJOBS=2
~/.local/bin/micromamba run -n align python3 convert_nd2_to_tiff.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --njobs "$NJOBS"
