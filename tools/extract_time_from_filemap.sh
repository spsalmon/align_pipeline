#!/bin/bash

#SBATCH -J extract_time
#SBATCH -o ./sbatch_output/extract_time.out
#SBATCH -e ./sbatch_output/extract_time.err
#SBATCH -c 32
#SBATCH -t 48:00:00
#SBATCH --mem=32GB

~/.local/bin/micromamba run -n align python3 extract_time_from_filemap.py
