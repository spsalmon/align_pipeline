import argparse
import os
import pickle
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np
import polars as pl
import yaml
from align_toolbox.foundation.file_handling import read_filemap, write_filemap
from align_toolbox.foundation.image_handling import get_acquisition_date
from joblib import Parallel, delayed, parallel_config

# ---- File handling ----


def backup_file(file_path, destination_dir):
    # Ensure the source file exists
    if not os.path.exists(file_path):
        print(f"Source file does not exist: {file_path}")
        return False

    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        print(
            f"Destination directory does not exist, attempting to create: {destination_dir}"
        )
        try:
            os.makedirs(destination_dir)
        except OSError as e:
            print(f"Failed to create destination directory: {e}")
            return False

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    file_extension = os.path.splitext(file_path)[1]
    destination_file_path = os.path.join(
        destination_dir, f"{base_name}{file_extension}"
    )

    # If a file with the same name exists, append an incrementing number
    i = 1
    while os.path.exists(destination_file_path):
        destination_file_path = os.path.join(
            destination_dir, f"{base_name}_{i}{file_extension}"
        )
        i += 1

    # Attempt to copy the file
    try:
        shutil.copyfile(file_path, destination_file_path)
        print(f"File backed up as: {destination_file_path}")
        return True
    except OSError as e:
        print(f"Failed to backup file: {e}")
        return False


def sync_backup_folder(dir, backup_dir):
    """
    Simple synchronization of a directory to a backup directory.
    Only copies files that don't exist or are older in the temp directory.
    """

    # Walk through backup directory
    for root, dirs, files in os.walk(dir):
        # Get the relative path
        rel_path = os.path.relpath(root, dir)
        backup_path = os.path.join(backup_dir, rel_path)

        # Create directory in temp if it doesn't exist
        os.makedirs(backup_path, exist_ok=True)

        # Copy each file if needed
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(backup_path, file)

            # Copy if destination doesn't exist or source is newer
            if not os.path.exists(dst_file) or os.path.getmtime(
                src_file
            ) > os.path.getmtime(dst_file):
                shutil.copy2(src_file, dst_file)


def get_experiment_subdirs(config):
    experiment_dir = config["experiment_dir"]
    raw_dir = config.get("raw_dir_name", "raw")
    raw_subdir = os.path.join(experiment_dir, raw_dir)
    subdirs = [
        f for f in os.listdir(raw_subdir) if os.path.isdir(os.path.join(raw_subdir, f))
    ]
    return sorted(subdirs)


def get_and_create_folders(config, subdir=None):
    experiment_dir = config["experiment_dir"]
    analysis_dir_name = config.get("analysis_dir_name", "analysis")
    raw_dir_name = config.get("raw_dir_name", "raw")

    raw_subdir = os.path.join(experiment_dir, raw_dir_name)
    analysis_subdir = os.path.join(experiment_dir, analysis_dir_name)
    os.makedirs(analysis_subdir, exist_ok=True)
    report_subdir = os.path.join(analysis_subdir, "report")
    os.makedirs(report_subdir, exist_ok=True)

    if subdir is not None:
        raw_subdir = os.path.join(raw_subdir, subdir)
        report_subdir = os.path.join(report_subdir, subdir)
        os.makedirs(report_subdir, exist_ok=True)

    # Run provenance is kept with the results it produced.
    pipeline_backup_dir = os.path.join(report_subdir, "pipeline_backup")
    os.makedirs(pipeline_backup_dir, exist_ok=True)

    return (
        experiment_dir,
        raw_subdir,
        analysis_subdir,
        report_subdir,
        pipeline_backup_dir,
    )


def get_groups(config):
    try:
        return config["groups"]
    except KeyError:
        return None


def get_filter_rule(groups, run_on_option):
    if (groups is not None) or (run_on_option is not None):
        return groups[run_on_option]
    else:
        return None


def filter_files_with_filter_rule(file_groups, filter_rule):
    if filter_rule is not None:
        if isinstance(file_groups[0], str):
            return [
                file_group
                for file_group in file_groups
                if filter_rule.lower() in file_group.lower()
            ]
        else:
            return [
                file_group
                for file_group in file_groups
                if all(filter_rule.lower() in file.lower() for file in file_group)
            ]
    else:
        return file_groups


def filter_files_of_group(files, config, run_on_option):
    groups = get_groups(config)
    filter_rule = get_filter_rule(groups, run_on_option)
    return filter_files_with_filter_rule(files, filter_rule)


def get_output_name(
    config,
    input_name,
    task_name,
    subdir=None,
    channels=None,
    return_subdir=True,
    add_raw=False,
    suffix=None,
    custom_suffix=None,
):
    analysis_subdir = config["analysis_subdir"]
    report_subdir = config["report_subdir"]
    raw_dir_name = config.get("raw_dir_name", "raw")
    analysis_dir_name = config.get("analysis_dir_name", "analysis")

    split = input_name.split("/")
    if len(split) > 1 and analysis_dir_name in split[0]:
        input_name = split[1:]
        input_name = os.path.join(*input_name)

    output_name = ""
    if channels is not None:
        if isinstance(channels, list):
            for channel in channels:
                output_name += f"ch{channel+1}_"
        else:
            output_name += f"ch{channels+1}_"
    # The literal "raw" and the configured raw_dir_name both denote the raw
    # input, which contributes no basename unless add_raw is set.
    if (input_name != raw_dir_name and input_name != "raw") or add_raw:
        output_name += os.path.basename(os.path.normpath(input_name)) + "_"
    output_name += task_name
    if suffix is not None:
        output_name += f"_{suffix}"
    if custom_suffix is not None:
        output_name += f"_{custom_suffix}"

    if return_subdir:
        output_name = os.path.join(analysis_subdir, output_name)
        if subdir is not None:
            output_name = os.path.join(output_name, subdir)
        os.makedirs(output_name, exist_ok=True)
    else:
        output_name = os.path.join(
            report_subdir, f"{output_name}.{config['report_format']}"
        )
    return output_name


def setup_run_dir(temp_dir, backend="slurm"):
    # Give the run its own directory under temp_dir (slurm job id, or start time
    # without slurm), create its folder structure and, on slurm, move the
    # launcher's logs in. Returns the run directory. The job id comes from
    # _init_pipeline.sh, not SLURM_JOB_ID, which is also set in any interactive
    # allocation and would make every run there share one directory.
    job_id = os.environ.get("ALIGN_INIT_JOB_ID")
    if not job_id:
        run_dir = os.path.join(
            temp_dir, datetime.now().strftime("pipeline_%Y%m%d-%H%M%S")
        )
    else:
        run_dir = os.path.join(temp_dir, f"pipeline_{job_id}")

    # batch/ and sbatch_output/ only hold generated job scripts and their slurm
    # logs, so they are pointless for a local run.
    os.makedirs(os.path.join(run_dir, "pickles"), exist_ok=True)
    if backend == "slurm":
        os.makedirs(os.path.join(run_dir, "batch"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "sbatch_output"), exist_ok=True)

    if not job_id:
        return run_dir

    log_dir = os.path.join(run_dir, "sbatch_output")
    try:
        # The launcher's -o/-e land next to the submit directory; bring them in
        # with the block logs and point slurm at their new home.
        moved = {}
        for stream, suffix in (("StdOut", "out"), ("StdErr", "err")):
            target = os.path.join(log_dir, f"init_pipeline-{job_id}.{suffix}")
            source = os.path.join("sbatch_output", f"pipeline-{job_id}.{suffix}")
            if os.path.exists(source):
                shutil.move(source, target)
            moved[stream] = target
        subprocess.run(
            ["scontrol", "update", f"JobId={job_id}"]
            + [f"{stream}={path}" for stream, path in moved.items()],
            check=False,
        )
    except Exception as e:
        print(f"Warning: could not move the launcher logs into {log_dir}: {e}")
    # The launcher's repo-root sbatch_output/ is left an empty landing zone;
    # cleanup_run() removes it at the end of a successful run.
    return run_dir


def cleanup_run(temp_dir):
    # End-of-run cleanup for a finished run: drop its whole temp dir (already
    # mirrored into the backup) and the launcher's now-empty repo-root
    # sbatch_output landing zone. Only the durable backup and the outputs remain.
    shutil.rmtree(temp_dir, ignore_errors=True)
    try:
        os.rmdir("sbatch_output")  # succeeds only if empty; absent in local runs
    except OSError:
        pass


def process_input_output_files(input_files, output_dir, rerun):
    try:
        input_files = list(input_files)
    except TypeError:
        return None, None

    if len(input_files) == 0:
        return None, None

    # Validate each element because input_files is a row-like container.
    if any(
        (file is None) or (not isinstance(file, str)) or (file.strip() == "")
        for file in input_files
    ):
        return None, None

    try:
        output_file = os.path.join(output_dir, os.path.basename(input_files[0].strip()))
    except Exception as e:
        print(f"Raised exception {e} for files {input_files}")
        return None, None

    if rerun or not os.path.exists(output_file):
        return input_files, output_file
    else:
        return None, None


def resolve_ref(ref, config):
    # Normalize a directory reference to the `{analysis_dir_name}/{name}` column,
    # so it can be written with or without the analysis-dir prefix (and survives
    # renaming analysis_dir_name). The literal "raw" and the configured
    # raw_dir_name both map to the raw column; absolute paths pass through.
    raw_dir_name = config.get("raw_dir_name", "raw")
    analysis_dir_name = config.get("analysis_dir_name", "analysis")
    # os.path.isabs misses POSIX absolute paths on Windows; a folder ref is never
    # a leading-slash relative name, so treat a leading "/" as absolute too.
    if os.path.isabs(ref) or ref.startswith("/"):
        return ref
    if ref == "raw" or ref == raw_dir_name:
        return raw_dir_name
    return f"{analysis_dir_name}/{os.path.basename(os.path.normpath(ref))}"


def get_input_and_output_files(experiment_filemap, columns, output_dir, rerun=True):
    all_input_files = [
        experiment_filemap.select(pl.col(column)).to_series().to_list()
        for column in columns
    ]
    all_input_files = np.array(all_input_files).T.tolist()
    with parallel_config(backend="loky", n_jobs=-1):
        results = Parallel()(
            delayed(process_input_output_files)(input_files, output_dir, rerun)
            for input_files in all_input_files
        )

    results = [result for result in results if result[0] is not None]
    input_files, output_files = zip(*results) if results else ([], [])

    input_files = list(input_files)
    output_files = list(output_files)

    return input_files, output_files


def get_experiment_time_from_filemap(experiment_filemap, config, recompute=False):
    experiment_filemap = experiment_filemap.clone()
    raw_dir_name = config.get("raw_dir_name", "raw")

    # When recompute=False, skip rows that already have a valid ExperimentTime.
    # Only process reference rows (global min Time from existing data, needed to
    # establish T0 per Point) plus the rows that are missing ExperimentTime.
    if (
        not recompute
        and "ExperimentTime" in experiment_filemap.columns
        and "Time" in experiment_filemap.columns
    ):
        has_exp_time = experiment_filemap["ExperimentTime"].is_not_null()

        if has_exp_time.all():
            return experiment_filemap["ExperimentTime"]

        rows_existing = experiment_filemap.filter(has_exp_time)
        rows_needing = experiment_filemap.filter(~has_exp_time)

        # Include global min-Time rows from existing data so that the recursive
        # call below can determine T0 per Point correctly.
        global_min_time = rows_existing["Time"].min()
        reference_rows = rows_existing.filter(pl.col("Time") == global_min_time)

        sub_filemap = (
            pl.concat([reference_rows, rows_needing])
            .unique(["Time", "Point"])
            .sort(["Time", "Point"])
        )

        sub_exp_time = get_experiment_time_from_filemap(
            sub_filemap, config, recompute=True
        )
        sub_filemap = sub_filemap.with_columns(sub_exp_time.alias("_exp_time"))

        # Keep only the newly computed values (exclude reference rows)
        new_computed = sub_filemap.join(
            reference_rows.select(["Time", "Point"]),
            on=["Time", "Point"],
            how="anti",
        ).select(["Time", "Point", "_exp_time"])

        result = (
            experiment_filemap.join(new_computed, on=["Time", "Point"], how="left")
            .with_columns(
                pl.when(pl.col("ExperimentTime").is_null())
                .then(pl.col("_exp_time"))
                .otherwise(pl.col("ExperimentTime"))
                .alias("ExperimentTime")
            )
            .drop("_exp_time")
        )
        return result["ExperimentTime"]

    raw_files = experiment_filemap.select(pl.col(raw_dir_name)).to_series().to_list()

    with parallel_config(backend="multiprocessing", n_jobs=-1):
        date_result = Parallel()(
            delayed(get_acquisition_date)(raw) for raw in raw_files
        )

    experiment_filemap = experiment_filemap.with_columns(
        pl.Series("date", date_result, dtype=pl.Datetime)
    )

    # Cast to datetime only if it's not already a datetime type
    if experiment_filemap.select(pl.col("date")).to_series().dtype != pl.Datetime:
        experiment_filemap = experiment_filemap.with_columns(
            pl.col("date").str.to_datetime(time_zone="UTC")
        )
    elif experiment_filemap.select(pl.col("date")).to_series().dtype.time_zone is None:
        experiment_filemap = experiment_filemap.with_columns(
            pl.col("date").dt.replace_time_zone("UTC")
        )

    # remove all timezone information, to prevent issues with some scopes
    if (
        experiment_filemap.select(pl.col("date")).to_series().dtype.time_zone
        is not None
    ):
        experiment_filemap = experiment_filemap.with_columns(
            pl.col("date").dt.replace_time_zone(None)
        )

    if experiment_filemap.select(pl.col("date")).to_series().null_count() == len(
        experiment_filemap
    ):
        return pl.Series("ExperimentTime", [None] * len(experiment_filemap))

    min_time = experiment_filemap.select(pl.col("Time")).to_series().min()
    try:
        first_time = (
            experiment_filemap.filter(pl.col("Time") == min_time)
            .group_by("Point", maintain_order=True)
            .agg(pl.col("date").first())
        )
    except Exception:
        print(
            "### Error: Time 0 not found for all points, experiment time will be computed from lowest Time value for each point.###"
        )
        first_time = (
            experiment_filemap.group_by("Point", maintain_order=True)
            .agg(pl.col("Time").arg_min().alias("min_idx"))
            .join(experiment_filemap, on="Point")
            .group_by("Point", maintain_order=True)
            .agg(pl.col("date").first())
        )

    # remove timezone from first_time if present
    if first_time.select(pl.col("date")).to_series().dtype.time_zone is not None:
        first_time = first_time.with_columns(pl.col("date").dt.replace_time_zone(None))

    first_time = first_time.rename({"date": "first_date"})

    experiment_filemap = experiment_filemap.join(first_time, on="Point", how="left")

    experiment_filemap = experiment_filemap.with_columns(
        ((pl.col("date") - pl.col("first_date")).dt.total_seconds().round(0)).alias(
            "ExperimentTime"
        )
    )

    return experiment_filemap.select(pl.col("ExperimentTime")).to_series()


# ---- Pickling ----


def load_pickles(*pickle_paths):
    loaded_pickles = []
    for pickle_path in pickle_paths:
        with open(pickle_path, "rb") as f:
            files = pickle.load(f)
        loaded_pickles.append(files)
    return loaded_pickles


def pickle_objects(temp_dir, *objects):
    pickled_paths = []
    for obj in objects:
        path = obj["path"]
        pickled_path = f"{os.path.join(temp_dir, 'pickles', path)}.pkl"

        if hasattr(obj["obj"], "to_pickle"):
            obj["obj"].to_pickle(pickled_path)
            pickled_paths.append(pickled_path)
        else:
            with open(pickled_path, "wb") as f:
                pickle.dump(obj["obj"], f)
            pickled_paths.append(pickled_path)
    return pickled_paths


def cleanup_files(*filepaths):
    for filepath in filepaths:
        try:
            os.remove(filepath)
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except PermissionError:
            print(f"Permission denied: {filepath}")
        except Exception as e:
            print(f"Error deleting file {filepath}: {e}")


# ---- SLURM ----

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_PIPELINE_DIR)


def resolve_slurm_config_path(global_config, config_file):
    # Where the SLURM resource file lives: an explicit slurm_config path
    # (relative ones resolved against the main config file's directory), else
    # the bundled default shipped alongside the example config.
    slurm_config_path = global_config.get("slurm_config")
    if slurm_config_path is None:
        return os.path.join(_PIPELINE_DIR, "defaults", "configs", "slurm_config.yaml")
    if not os.path.isabs(slurm_config_path):
        return os.path.join(
            os.path.dirname(os.path.abspath(config_file)), slurm_config_path
        )
    return slurm_config_path


def merge_slurm_config(global_config, config_file):
    # SLURM resource requests live in a separate file for the cluster backend,
    # so the cluster knobs sit in one place. Merge them into the config at
    # startup; inline sbatch_* keys take precedence, so single-file configs keep
    # working. No-op for the local backend.
    if global_config.get("backend", "slurm") != "slurm":
        return global_config

    slurm_config_path = resolve_slurm_config_path(global_config, config_file)
    if not os.path.exists(slurm_config_path):
        print(f"SLURM config file not found, skipping: {slurm_config_path}")
        return global_config

    with open(slurm_config_path) as f:
        slurm_config = yaml.load(f, Loader=yaml.FullLoader) or {}

    # Inline keys win over the file.
    for key, value in slurm_config.items():
        global_config.setdefault(key, value)
    return global_config


_SBATCH_KEYS = (
    "sbatch_cpus",
    "sbatch_time",
    "sbatch_memory",
    "sbatch_gpus",
    "sbatch_extra_options",
)


def _slurm_defaults(config):
    # The shared resource default: the top-level sbatch_* keys.
    return {key: config[key] for key in _SBATCH_KEYS if key in config}


def _merge_slurm_section(resolved, section):
    # Scalar sbatch_* keys of the section override the defaults, but the
    # sbatch_extra_options list accumulates, so a section cannot silently drop a
    # cluster-wide entry (e.g. --account).
    extra = list(resolved.get("sbatch_extra_options") or []) + list(
        section.get("sbatch_extra_options") or []
    )
    resolved.update(section)
    if extra:
        resolved["sbatch_extra_options"] = extra
    return resolved


def resolve_block_slurm(config, block_name):
    # Effective SLURM resources for a worker block: the shared defaults overlaid
    # with any per-type entry under sbatch_overrides.
    resolved = _slurm_defaults(config)
    return _merge_slurm_section(
        resolved, config.get("sbatch_overrides", {}).get(block_name, {})
    )


def resolve_init_slurm(config):
    # Effective SLURM resources for the outer/orchestrator job: the shared
    # defaults (minus the GPU, which the init job never needs) overlaid with
    # sbatch_init.
    resolved = {
        key: value
        for key, value in _slurm_defaults(config).items()
        if key != "sbatch_gpus"
    }
    return _merge_slurm_section(resolved, config.get("sbatch_init", {}))


def build_resource_directives(cores, time_limit, memory, gpus, extra_options):
    # sbatch resource options (no job name/output), shared by the per-block
    # script header and the outer job's CLI flags. Each standard directive is
    # emitted only when set, so a cluster can drop one (e.g. omit --mem in favour
    # of a --mem-per-cpu entry in extra_options); extra_options are raw sbatch
    # option strings used verbatim (e.g. "--account=gratis").
    directives = []
    if cores is not None:
        directives.append(f"-c {cores}")
    if time_limit is not None:
        directives.append(f"-t {time_limit}")
    if memory is not None:
        directives.append(f"--mem={memory}")
    if gpus is not None:
        directives.append(f"--gres=gpu:{gpus}")
    for option in extra_options or []:
        directives.append(str(option))
    return directives


def backup_run_config(global_config, config_file, temp_dir):
    # Snapshot the config file(s) actually used into the run's temp dir (synced
    # into the pipeline backup) as a write-only record of the run. Copies the
    # main config, plus the resolved SLURM config for the cluster backend.
    shutil.copy2(config_file, temp_dir)
    if global_config.get("backend", "slurm") == "slurm":
        slurm_config_path = resolve_slurm_config_path(global_config, config_file)
        if os.path.exists(slurm_config_path):
            shutil.copy2(slurm_config_path, temp_dir)


def save_version_control_info(temp_dir):
    # Record git branch/commit/status + interpreter/package versions into the
    # run's temp dir (synced to the backup) so a run stays reproducible.
    lines = []
    try:
        for label, rev in (("Git Branch", "--abbrev-ref"), ("Git Commit", "")):
            out = subprocess.run(
                ["git", "-C", _REPO_DIR, "rev-parse", *([rev] if rev else []), "HEAD"],
                capture_output=True,
                text=True,
            )
            lines.append(f"{label}: {out.stdout.strip()}")
        status = subprocess.run(
            ["git", "-C", _REPO_DIR, "status"], capture_output=True, text=True
        )
        lines.append("Git Status:\n" + status.stdout.strip())
    except Exception as e:
        lines.append(f"Version control info unavailable: {e}")

    lines.append(f"Python Version: {sys.version}")
    try:
        import align_toolbox

        lines.append(f"align_toolbox Version: {align_toolbox.__version__}")
    except Exception as e:
        lines.append(f"align_toolbox Version unavailable: {e}")

    with open(os.path.join(temp_dir, "git_info.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def get_python_command(config):
    # Prefix to launch python workers: `python_command` if set, else the active
    # interpreter (local) or the default micromamba runner (slurm).
    python_command = config.get("python_command")
    if python_command:
        return python_command
    if config.get("backend", "slurm") == "local":
        return sys.executable
    return "~/.local/bin/micromamba run -n align python3"


def create_linker_command(
    python_command,
    temp_dir,
    result=None,
):
    # A block with no output has no result to record, so --result is left off
    # rather than passed as the string "None".
    linker_command = (
        f"{python_command} -m align_pipeline.block_linker --temp_dir {temp_dir}"
    )
    if result is not None:
        linker_command += f" --result {result}"
    return linker_command


def concatenate_sbatch_logs(temp_dir, closing=""):
    # Join the per-block slurm logs into one file per stream in the run's temp
    # dir, ordered by write time (blocks run sequentially). Originals are kept
    # and the combined files are rebuilt on every call, so a run that stops
    # mid-chain still leaves a readable log. `closing` is the last line, saying
    # whether the run got to the end or is still going.
    log_dir = os.path.join(temp_dir, "sbatch_output")
    if not os.path.isdir(log_dir):
        return
    job_id = os.path.basename(os.path.normpath(temp_dir)).removeprefix("pipeline_")
    try:
        for suffix in (".out", ".err"):
            parts = sorted(
                (
                    os.path.join(log_dir, name)
                    for name in os.listdir(log_dir)
                    if name.endswith(suffix)
                ),
                key=os.path.getmtime,
            )
            if not parts:
                continue
            # Written to the parent, so a rerun never reads its own output back.
            with open(os.path.join(temp_dir, f"pipeline-{job_id}{suffix}"), "w") as out:
                for part in parts:
                    out.write(f"===== {os.path.basename(part)} =====\n")
                    with open(part, errors="replace") as f:
                        shutil.copyfileobj(f, out)
                    out.write("\n")
                out.write(f"===== {closing} =====\n")
    except Exception as e:
        # Log housekeeping must never take the pipeline down with it.
        print(f"Warning: could not combine the sbatch logs: {e}")


def block_label(building_blocks, index):
    # Short "3/12 straightening" tag identifying one entry of the block sequence,
    # used by the progress prints (blocks of the same type repeat, so the
    # position is what tells them apart).
    entry = building_blocks[index]
    label = f"{index + 1}/{len(building_blocks)} {entry['block'].name}"
    if entry["subdir"] is not None:
        label += f" [{entry['subdir']}]"
    return label


def run_command_local(command, run_linker=True, linker_command=None):
    # Run a block's command synchronously, then chain to the linker (next block).
    # Skip empty / commented-out commands (a block with no input files to process).
    stripped = command.strip()
    if stripped and not stripped.startswith("#"):
        subprocess.run(command, shell=True, check=True)
    if run_linker and linker_command is not None:
        subprocess.run(linker_command, shell=True, check=True)


def run_command(
    command,
    script_name,
    config,
    run_linker=True,
    linker_command=None,
    requires_gpu=False,
):
    # Local backend: run the worker (and linker) directly instead of submitting to slurm.
    if config.get("backend", "slurm") == "local":
        run_command_local(command, run_linker=run_linker, linker_command=linker_command)
        return

    # Per-block resources: shared defaults overlaid with this block type's
    # sbatch_overrides entry.
    block_slurm = resolve_block_slurm(config, script_name)

    # GPU directive only for blocks that need it and when a gpu is configured.
    gpus = block_slurm.get("sbatch_gpus")
    if not requires_gpu:
        gpus = None

    # Cores requested follow n_jobs when sbatch_cpus is unset (and vice versa).
    cores = block_slurm.get("sbatch_cpus", config.get("n_jobs"))

    script_path = create_sbatch_file(
        script_name,
        config["temp_dir"],
        cores,
        block_slurm.get("sbatch_time"),
        block_slurm.get("sbatch_memory"),
        command,
        gpus=gpus,
        extra_options=block_slurm.get("sbatch_extra_options"),
        run_linker=run_linker,
        linker_command=linker_command,
    )
    # Give sbatch's own "Submitted batch job <id>" line some context.
    print(f"Submitting {script_name} to slurm ...", flush=True)
    subprocess.run(["sbatch", script_path])


def create_sbatch_file(
    job_name,
    temp_dir,
    cores,
    time_limit,
    memory,
    command,
    gpus=None,
    extra_options=None,
    run_linker=True,
    linker_command=None,
):
    # Ensure batch directory exists
    batch_dir = os.path.join(temp_dir, "batch")
    os.makedirs(batch_dir, exist_ok=True)

    # Build SLURM header: job name + output paths, then the resource directives.
    directives = [
        f"-J {job_name}",
        f"-o {os.path.join(temp_dir, 'sbatch_output', job_name)}-%j.out",
        f"-e {os.path.join(temp_dir, 'sbatch_output', job_name)}-%j.err",
    ]
    directives += build_resource_directives(
        cores, time_limit, memory, gpus, extra_options
    )

    header = "#!/bin/bash\n" + "".join(f"#SBATCH {d}\n" for d in directives)

    content = f"""{header}
## this is a test for removing issues with lock files
# export TMPDIR={os.path.join(temp_dir, 'tmp', "$SLURM_JOB_ID")}
# mkdir -p $TMPDIR
# export XDG_CACHE_HOME={os.path.join(temp_dir, 'cache')}
"""

    # set environment variables for single threaded execution (doesn't solve our problem, so I commented it out)
    #     content += """
    # export OMP_NUM_THREADS=1
    # export MKL_NUM_THREADS=1
    # export OPENBLAS_NUM_THREADS=1
    # """
    content += "\n" + command + "\n"

    if run_linker and linker_command is not None:
        content += linker_command + "\n"

    script_path = os.path.join(batch_dir, f"{job_name}.sh")
    with open(script_path, "w") as file:
        file.write(content)

    return script_path


# ---- Command-line interface ----


def basic_get_args() -> argparse.Namespace:
    """
    Parses the command-line arguments and returns them as a namespace object.

    Returns:
        argparse.Namespace: The namespace object containing the parsed arguments.
    """
    # Create a parser and set the formatter class to ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        description="Read args for a piece of the pipeline."
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input file paths (saved in a pickle file) or single filepath (CSV file for example).",
    )
    parser.add_argument("-o", "--output", help="Output file path or pickle.")
    parser.add_argument(
        "-b",
        "--block_config",
        help="Pickled configuration of the current block's (dictionary)",
    )
    parser.add_argument("-c", "--config", help="Pickled FULL config dictionary.")
    parser.add_argument(
        "-j", "--n_jobs", type=int, help="Number of jobs for parallel execution."
    )
    parser.add_argument(
        "-f", "--filemap", help="Pickled experiment filemap (if needed).", default=None
    )

    return parser.parse_args()


# ---- Saving ----


def rename_merge_and_save_records(
    experiment_filemap,
    filemap_path,
    records_file,
    column_name_old,
    column_name_new,
    merge_cols=["Time", "Point"],
):
    dataframe = read_filemap(records_file)

    dataframe = dataframe.rename({column_name_old: column_name_new})

    if column_name_new in experiment_filemap.columns:
        experiment_filemap = experiment_filemap.drop(column_name_new)

    experiment_filemap = experiment_filemap.join(dataframe, on=merge_cols, how="left")

    write_filemap(experiment_filemap, filemap_path)

    return experiment_filemap


def merge_and_save_records(
    experiment_filemap, filemap_path, records_file, merge_cols=["Time", "Point"]
):
    dataframe = read_filemap(records_file)

    new_columns = [column for column in dataframe.columns if column not in merge_cols]

    # Drop any overlapping columns from experiment_filemap
    columns_to_drop = [col for col in new_columns if col in experiment_filemap.columns]
    if columns_to_drop:
        experiment_filemap = experiment_filemap.drop(columns_to_drop)

    experiment_filemap = experiment_filemap.join(dataframe, on=merge_cols, how="left")
    write_filemap(experiment_filemap, filemap_path)

    return experiment_filemap
