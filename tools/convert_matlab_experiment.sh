#!/bin/bash

#SBATCH -J convert_matlab
#SBATCH -o ../sbatch_output/convert_matlab.out
#SBATCH -e ../sbatch_output/convert_matlab.err
#SBATCH -c 32
#SBATCH -t 48:00:00
#SBATCH --mem=32GB

~/.local/bin/micromamba run -n align python3 convert_matlab_experiment.py
