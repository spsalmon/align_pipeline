"""Pipeline entry point: load the config, build the sequence of building blocks
for the experiment (per raw subdir), and launch the first one. The block linker
chains the rest. Exposed as the `align` command via main().
"""

import argparse
import os

import numpy as np
import polars as pl
import yaml
from align_toolbox.foundation.file_handling import (
    get_dir_filemap,
    read_filemap,
    write_filemap,
)

from align_pipeline.building_blocks import (
    parse_and_create_building_blocks,
    validate_config,
)
from align_pipeline.utils import (
    backup_run_config,
    block_label,
    get_and_create_folders,
    get_experiment_subdirs,
    get_experiment_time_from_filemap,
    merge_slurm_config,
    pickle_objects,
    save_version_control_info,
    setup_run_dir,
    sync_backup_folder,
)


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    # The config can be given positionally or with -c; -c wins if both are set.
    parser.add_argument("config_positional", nargs="?", help="Path to the config file")
    parser.add_argument("-c", "--config", help="Path to the config file")
    parser.add_argument(
        "-t",
        "--temp_dir",
        help="Path to the directory storing temporary files (overrides the config)",
        required=False,
    )
    parser.add_argument(
        "-e",
        "--experiment_dir",
        help="Path to the experiment directory (overrides the config)",
        required=False,
    )
    args = parser.parse_args(argv)
    args.config = args.config or args.config_positional
    if not args.config:
        parser.error("a config file is required (positional or -c/--config)")
    return args


SUPPORTED_FILEMAP_FORMATS = ("csv", "parquet")


def _file_creation_time(path):
    # st_birthtime is not exposed on Linux; fall back to the modification time.
    stat = os.stat(path)
    return getattr(stat, "st_birthtime", stat.st_mtime)


def find_existing_filemap(report_subdir, filemap_name):
    """Return the path of an existing `<filemap_name>.<ext>` in report_subdir for
    any supported extension, or None. If several exist (e.g. both a .csv and a
    .parquet), the most recently created one wins."""
    candidates = [
        os.path.join(report_subdir, f"{filemap_name}.{ext}")
        for ext in SUPPORTED_FILEMAP_FORMATS
    ]
    candidates = [path for path in candidates if os.path.exists(path)]
    if not candidates:
        return None
    return max(candidates, key=_file_creation_time)


def build_or_load_filemap(config):
    """Build the base filemap from the raw dir (or load the existing one) and add
    ExperimentTime. Returns (experiment_filemap, filemap_path)."""
    report_subdir = config["report_subdir"]
    report_format = config["report_format"]
    raw_subdir = config["raw_subdir"]
    raw_dir_name = config["raw_dir_name"]
    time_regex = config.get("time_regex", r"Time(\d+)")
    point_regex = config.get("point_regex", r"Point(\d+)")
    extract_experiment_time = config.get("get_experiment_time", True)
    overwrite_annotated = config.get("overwrite_annotated_filemap", False)

    # if no filemap exists (in any supported format), create it from the raw directory
    base_filemap_path = find_existing_filemap(report_subdir, "analysis_filemap")
    if base_filemap_path is None:
        try:
            experiment_filemap = get_dir_filemap(raw_subdir, time_regex, point_regex)
        except Exception as e:
            print(f"Error generating filemap from directory: {e}")
            experiment_filemap = pl.DataFrame()

        # if the filemap is empty, it's probably because they do not follow the Time, Point structure
        if experiment_filemap.is_empty():
            image_paths = sorted(
                [os.path.join(raw_subdir, f) for f in os.listdir(raw_subdir)]
            )
            experiment_filemap = pl.DataFrame({"ImagePath": image_paths})
            config["no_timepoints"] = True

        experiment_filemap = experiment_filemap.rename({"ImagePath": raw_dir_name})
        base_filemap_path = os.path.join(
            report_subdir, f"analysis_filemap.{report_format}"
        )
        write_filemap(experiment_filemap, base_filemap_path)

    # select the right filemap
    annotated_filemap_path = find_existing_filemap(
        report_subdir, "analysis_filemap_annotated"
    )
    if overwrite_annotated and annotated_filemap_path is not None:
        source_filemap_path = annotated_filemap_path
        filemap_name = "analysis_filemap_annotated"
    else:
        source_filemap_path = base_filemap_path
        filemap_name = "analysis_filemap"

    # The run always works on the filemap in the configured report_format; one
    # found in another format is converted and left untouched.
    filemap_path = os.path.join(report_subdir, f"{filemap_name}.{report_format}")
    experiment_filemap = read_filemap(source_filemap_path)
    if source_filemap_path != filemap_path:
        print(
            f"### Converting {source_filemap_path} to {report_format}: {filemap_path} ###"
        )
        write_filemap(experiment_filemap, filemap_path)

    # Check for new files added to the raw directory since the filemap was created
    if "Time" in experiment_filemap.columns and "Point" in experiment_filemap.columns:
        try:
            current_dir_filemap = get_dir_filemap(raw_subdir, time_regex, point_regex)
            if not current_dir_filemap.is_empty():
                current_dir_filemap = current_dir_filemap.rename(
                    {"ImagePath": raw_dir_name}
                )
                new_rows = current_dir_filemap.join(
                    experiment_filemap.select(["Time", "Point"]),
                    on=["Time", "Point"],
                    how="anti",
                )
                if not new_rows.is_empty():
                    print(
                        f"### Found {len(new_rows)} new file(s) in {raw_subdir}, adding to filemap ###"
                    )
                    for col in experiment_filemap.columns:
                        if col not in new_rows.columns:
                            new_rows = new_rows.with_columns(
                                pl.lit(None)
                                .cast(experiment_filemap[col].dtype)
                                .alias(col)
                            )
                    new_rows = new_rows.select(experiment_filemap.columns)
                    experiment_filemap = pl.concat([experiment_filemap, new_rows]).sort(
                        ["Time", "Point"]
                    )
                    write_filemap(experiment_filemap, filemap_path)
        except Exception as e:
            print(f"Warning: could not check for new files in raw directory: {e}")

    config["filemap_path"] = filemap_path

    # Compute ExperimentTime if missing. When rows already have it (e.g. existing
    # filemap), only compute for the new rows that lack it (recompute=False).
    has_missing_exp_time = (
        "ExperimentTime" in experiment_filemap.columns
        and "Time" in experiment_filemap.columns
        and experiment_filemap["ExperimentTime"].is_null().any()
    )
    # ExperimentTime needs a Time column; without one (no_timepoints filemap)
    # skip extraction and leave it null instead of crashing.
    can_compute_exp_time = (
        extract_experiment_time and "Time" in experiment_filemap.columns
    )
    if "ExperimentTime" not in experiment_filemap.columns or has_missing_exp_time:
        if can_compute_exp_time:
            print("### Calculating ExperimentTime ###")
            experiment_filemap = experiment_filemap.with_columns(
                pl.lit(
                    get_experiment_time_from_filemap(
                        experiment_filemap, config, recompute=not has_missing_exp_time
                    )
                ).alias("ExperimentTime")
            )
            write_filemap(experiment_filemap, filemap_path)
        else:
            experiment_filemap = experiment_filemap.with_columns(
                pl.lit(np.nan).alias("ExperimentTime")
            )
            write_filemap(experiment_filemap, filemap_path)

    return experiment_filemap, filemap_path


def build_blocks_for_subdir(global_config, temp_dir_basename, temp_dir, subdir=None):
    print(f"### Initializing the pipeline for subdir {subdir} ###")

    (
        _,
        raw_subdir,
        analysis_subdir,
        report_subdir,
        pipeline_backup_dir,
    ) = get_and_create_folders(global_config, subdir)

    pipeline_backup_dir = os.path.join(pipeline_backup_dir, temp_dir_basename)
    os.makedirs(pipeline_backup_dir, exist_ok=True)

    config = global_config.copy()

    config["raw_subdir"] = raw_subdir
    if subdir is None:
        raw_dir_name = os.path.basename(os.path.normpath(raw_subdir))
    else:
        # instead, get the name from the parent directory
        raw_dir_name = os.path.basename(os.path.normpath(os.path.dirname(raw_subdir)))

    config["raw_dir_name"] = raw_dir_name
    config["analysis_subdir"] = analysis_subdir
    config["report_subdir"] = report_subdir
    config["pipeline_backup_dir"] = pipeline_backup_dir
    config["temp_dir"] = temp_dir

    report_format = config.get("report_format", "csv")
    config["report_format"] = report_format

    sync_backup_folder(temp_dir, pipeline_backup_dir)

    # If construction of a filemap we created this run fails partway, drop it so
    # the next run rebuilds instead of reusing the partial one (regen gates on absence).
    base_filemap_path = os.path.join(report_subdir, f"analysis_filemap.{report_format}")
    already_existed = (
        find_existing_filemap(report_subdir, "analysis_filemap") is not None
    )
    try:
        build_or_load_filemap(config)
    except Exception:
        if not already_existed and os.path.exists(base_filemap_path):
            os.remove(base_filemap_path)
        raise

    print("Building the config of the building blocks ...")

    building_blocks = parse_and_create_building_blocks(config)

    building_blocks = [
        {"block": block, "subdir": subdir, "config": config}
        for block in building_blocks
    ]
    print(f"Building blocks created: {len(building_blocks)}")

    return building_blocks


def main(argv=None):
    args = get_args(argv)
    config_file = args.config
    temp_dir = args.temp_dir

    with open(config_file) as f:
        global_config = yaml.load(f, Loader=yaml.FullLoader)

    # A CLI experiment_dir overrides the one in the config.
    if args.experiment_dir:
        global_config["experiment_dir"] = os.path.abspath(args.experiment_dir)

    # Pull SLURM resources in from their separate file (cluster backend only).
    global_config = merge_slurm_config(global_config, config_file)

    # Fail fast on a bad config, before any run dir or job is created.
    validate_config(global_config)

    # Temp files: --temp_dir, else a temp_dir config key, else ./temp_files in the
    # working directory (transient, gitignored; matches the sbatch launcher).
    if temp_dir:
        temp_dir = os.path.abspath(temp_dir)
    elif global_config.get("temp_dir"):
        temp_dir = os.path.abspath(global_config["temp_dir"])
    else:
        temp_dir = os.path.abspath("temp_files")

    # Each run gets its own subdirectory, so repeated runs never overwrite one
    # another's temp files or their backup.
    temp_dir = setup_run_dir(temp_dir, global_config.get("backend", "slurm"))
    temp_dir_basename = os.path.basename(temp_dir)

    # Snapshot the config(s) and version info into the run's temp dir (synced to
    # the backup) as a write-only record of the run.
    backup_run_config(global_config, config_file, temp_dir)
    save_version_control_info(temp_dir)

    subdirs = get_experiment_subdirs(global_config)
    building_blocks = []
    print(f"Running the pipeline for subdirs: {subdirs}")
    if not subdirs:
        building_blocks.extend(
            build_blocks_for_subdir(global_config, temp_dir_basename, temp_dir)
        )
    else:
        for subdir in subdirs:
            building_blocks.extend(
                build_blocks_for_subdir(
                    global_config, temp_dir_basename, temp_dir, subdir
                )
            )

    # initialize on 1 as we're gonna run the first block immediately
    progress_tracker = {"current_block_index": 1, "building_blocks": building_blocks}
    progress_tracker_pickle = {"path": "progress_tracker", "obj": progress_tracker}

    _ = pickle_objects(temp_dir, progress_tracker_pickle)

    # The planned sequence, so the per-block logs further down can be placed.
    print(f"### Pipeline plan: {len(building_blocks)} blocks ###")
    for index in range(len(building_blocks)):
        print(f"  {block_label(building_blocks, index)}")

    # Run the first building block
    current = building_blocks[0]
    current_building_block, current_subdir, current_config = (
        current["block"],
        current["subdir"],
        current["config"],
    )

    experiment_filemap = read_filemap(current_config["filemap_path"])
    print(f"### Starting block {block_label(building_blocks, 0)} ###")
    print(f"Running {current_building_block} ...", flush=True)
    current_building_block.run(
        experiment_filemap, current_config, subdir=current_subdir
    )


if __name__ == "__main__":
    main()
