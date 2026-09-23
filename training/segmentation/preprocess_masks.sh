#!/bin/bash

#SBATCH -J preprocess
#SBATCH -o ../../sbatch_output/preprocess.out
#SBATCH -e ../../sbatch_output/preprocess.err
#SBATCH -c 32
#SBATCH -t 1:00:00
#SBATCH --mem=32GB

DATABASE_PATH="/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/datasets/annas_weird_squid_experiment/pharynx/"
PREPROCESSING_TYPE="binarize"  # Change this to the desired preprocessing type
KEEP_ONLY_BIGGEST_OBJECT=true

if [ "$KEEP_ONLY_BIGGEST_OBJECT" = true ]; then
    ~/.local/bin/micromamba run -n align python3 preprocess_masks.py --database_path "$DATABASE_PATH" --preprocessing_type "$PREPROCESSING_TYPE" --keep_only_biggest_object
else
    ~/.local/bin/micromamba run -n align python3 preprocess_masks.py --database_path "$DATABASE_PATH" --preprocessing_type "$PREPROCESSING_TYPE"
fi
