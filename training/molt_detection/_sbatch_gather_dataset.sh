#!/bin/bash

#SBATCH -J gather_molt
#SBATCH -o ../../sbatch_output/gather_molt-%j.out
#SBATCH -e ../../sbatch_output/gather_molt-%j.err
#SBATCH -c 16
#SBATCH -t 24:00:00
#SBATCH --mem=32GB

# Default configuration file
DEFAULT_CONFIG_FILE="./configs/molt_dataset_config.yaml"
CONFIG_FILE="$DEFAULT_CONFIG_FILE"

# Function to show usage
usage() {
    echo "Usage: $0 [-c <config_file> | --config <config_file>]" >&2
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -c|--config)
        CONFIG_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        usage
        ;;
    esac
done

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file not found: $CONFIG_FILE" >&2
    exit 1
fi

# Run the Python script with the specified or default configuration file
~/.local/bin/micromamba run -n align_pipeline python3 get_molt_detection_data.py -c "$CONFIG_FILE"
