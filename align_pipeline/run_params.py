"""Emit run parameters for the shell launchers, derived from the pipeline
config so the launch scripts stay generic. Currently prints the sbatch CLI
flags for the outer/orchestrator job (from sbatch_init).
"""

import argparse
import os
import sys

import yaml

from align_pipeline.building_blocks import validate_config
from align_pipeline.utils import (
    build_resource_directives,
    merge_slurm_config,
    resolve_init_slurm,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    parser.add_argument(
        "-e",
        "--experiment_dir",
        help="Path to the experiment directory (overrides the config)",
        required=False,
    )
    parser.add_argument(
        "--sbatch-init",
        action="store_true",
        help="Print the sbatch CLI flags for the outer/orchestrator job",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the config; exit non-zero and print the errors if invalid",
    )
    return parser.parse_args()


def sbatch_init_flags(config):
    init = resolve_init_slurm(config)
    return build_resource_directives(
        init.get("sbatch_cpus"),
        init.get("sbatch_time"),
        init.get("sbatch_memory"),
        init.get("sbatch_gpus"),
        init.get("sbatch_extra_options"),
    )


def main():
    args = get_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Mirror init_pipeline: apply the -e override, then merge the slurm config,
    # so validation sees exactly what the run will.
    if args.experiment_dir:
        config["experiment_dir"] = os.path.abspath(args.experiment_dir)
    config = merge_slurm_config(config, args.config)

    if args.validate:
        try:
            validate_config(config)
        except ValueError as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    if args.sbatch_init:
        print(" ".join(sbatch_init_flags(config)))


if __name__ == "__main__":
    main()
