import os
import re
import shutil

import pandas as pd
from align_toolbox.foundation.file_handling import get_dir_filemap
from joblib import Parallel, delayed
from tqdm import tqdm

experiment_dir = "/mnt/towbin.data/shared/spsalmon/20240524_161257_273_LIPSI_40x_397_405_no_crash/analysis/nuclei_types"
dir_list = [
    "pad1",
    "pad2",
]
time_regex = r"Time(\d+)"
point_regex = r"Point(\d+)"

dir_list = [os.path.join(experiment_dir, d) for d in dir_list]
output_dir = os.path.join(experiment_dir, "")
backup_dir = os.path.join(experiment_dir, "backup")
os.makedirs(backup_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


def replace_time_in_filename(filename, new_time, time_regex=r"Time(\d+)"):
    match = re.search(time_regex, filename)
    if not match:
        return ""
    original_time_str = match.group(1)
    original_prefix = match.group(0)[: match.start(1) - match.start(0)]
    original_digit_count = len(original_time_str)
    new_time_str = f"{new_time:0{original_digit_count}d}"
    new_filename = re.sub(time_regex, f"{original_prefix}{new_time_str}", filename)
    return new_filename


def concat_timelapse_experiments(first_filemap, second_filemap):
    last_time = first_filemap["Time"].max()
    second_filemap = second_filemap.copy()  # avoid mutating the original
    second_filemap["Time"] = second_filemap["Time"] + last_time + 1

    first_filemap = first_filemap[first_filemap["ImagePath"].notna()]
    second_filemap = second_filemap[second_filemap["ImagePath"].notna()]

    combined_filemap = pd.concat([first_filemap, second_filemap], ignore_index=True)
    combined_filemap.reset_index(drop=True, inplace=True)
    combined_filemap["NewFilename"] = combined_filemap.apply(
        lambda row: os.path.basename(
            replace_time_in_filename(row["ImagePath"], row["Time"])
        ),
        axis=1,
    )
    combined_filemap.sort_values(by=["Point", "Time"], inplace=True)
    return combined_filemap


# ---------------------------------------------------------------------------
# Safety checks – call these before touching any files
# ---------------------------------------------------------------------------


def check_no_empty_new_filenames(combined_filemap):
    """Fail if any source filename had no Time\\d+ pattern and produced an empty NewFilename."""
    bad = combined_filemap[
        (combined_filemap["NewFilename"] == "") & (combined_filemap["ImagePath"] != "")
    ]
    if not bad.empty:
        raise ValueError(
            f"The following {len(bad)} file(s) produced an empty NewFilename "
            f"(no 'Time<digits>' pattern found in their path). "
            f"They would be lost or cause an OS error:\n"
            + bad[["ImagePath", "Time", "Point"]].to_string()
        )


def check_no_duplicate_output_filenames(combined_filemap):
    """Fail if two different source files would be moved to the same destination.
    Without this check, shutil.move silently overwrites the first file with the second.
    """
    dupes = combined_filemap[
        combined_filemap.duplicated(subset=["NewFilename"], keep=False)
    ]
    dupes = dupes[dupes["ImagePath"] != ""]  # ignore rows with missing source paths
    if not dupes.empty:
        raise ValueError(
            f"The following {len(dupes)} rows share an output filename. "
            f"Moving them would silently destroy data:\n"
            + dupes[["ImagePath", "NewFilename", "Time", "Point"]].to_string()
        )


def check_no_duplicate_source_paths(combined_filemap):
    """Fail if the same source file appears more than once (e.g. overlapping dir_list entries)."""
    dupes = combined_filemap[
        combined_filemap.duplicated(subset=["ImagePath"], keep=False)
    ]
    dupes = dupes[dupes["ImagePath"] != ""]  # ignore rows with missing source paths
    if not dupes.empty:
        raise ValueError(
            f"The following {len(dupes)} rows have duplicate source ImagePaths. "
            f"This suggests overlapping input directories:\n"
            + dupes[["ImagePath", "NewFilename", "Time", "Point"]].to_string()
        )


def check_source_files_exist(combined_filemap):
    """Fail fast if any source file is already missing."""
    missing = [
        p for p in combined_filemap["ImagePath"] if p != "" and not os.path.isfile(p)
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} source file(s) do not exist on disk. First 10:\n"
            + "\n".join(missing[:10])
        )


def check_output_files_do_not_exist(combined_filemap):
    """Warn (not fail) if an output path already exists – could indicate a partial previous run."""
    existing = [
        p
        for p in combined_filemap["OutputPath"]
        if p != "" and os.path.exists(p) and not os.path.isdir(p)
    ]
    if existing:
        raise FileExistsError(
            f"{len(existing)} output path(s) already exist. "
            f"Re-running would silently overwrite them. "
            f"Clear '{output_dir}' or resume from the saved CSV. First 10:\n"
            + "\n".join(existing[:10])
        )


def check_file_counts_consistent(dir_list, combined_filemap):
    """Fail if the total number of files in combined_filemap doesn't match
    the total number of image files found across all input directories.
    A mismatch means some files were silently dropped (e.g. by notna() filtering).
    """
    total_source = sum(
        get_dir_filemap(d, time_regex=time_regex, point_regex=point_regex)
        .to_pandas()["ImagePath"]
        .notna()
        .sum()
        for d in dir_list
    )
    total_combined = len(combined_filemap)
    if total_source != total_combined:
        raise ValueError(
            f"File count mismatch: {total_source} source files found across input dirs, "
            f"but combined_filemap has {total_combined} rows. "
            f"Some files may have been silently dropped."
        )


def run_all_safety_checks(dir_list, combined_filemap):
    print("Running pre-flight safety checks …")
    check_no_empty_new_filenames(combined_filemap)
    print("  ✓ No empty NewFilenames")
    check_no_duplicate_output_filenames(combined_filemap)
    print("  ✓ No duplicate output filenames")
    check_no_duplicate_source_paths(combined_filemap)
    print("  ✓ No duplicate source paths")
    check_source_files_exist(combined_filemap)
    print("  ✓ All source files exist on disk")
    check_output_files_do_not_exist(combined_filemap)
    print("  ✓ No output files already exist")
    check_file_counts_consistent(dir_list, combined_filemap)
    print("  ✓ File counts consistent")
    print("All checks passed – safe to proceed.\n")


# ---------------------------------------------------------------------------
# Main combine + move logic
# ---------------------------------------------------------------------------


def combine_all_and_move(dir_list, output_dir):
    filemaps = [
        get_dir_filemap(d, time_regex=time_regex, point_regex=point_regex)
        for d in dir_list
    ]
    filemaps = [fm.to_pandas() for fm in filemaps]

    combined_filemap = filemaps[0]
    for next_filemap in filemaps[1:]:
        combined_filemap = concat_timelapse_experiments(combined_filemap, next_filemap)

    combined_filemap["OutputPath"] = combined_filemap.apply(
        lambda row: os.path.join(output_dir, row["NewFilename"]), axis=1
    )
    combined_filemap.sort_values(by=["Point", "Time"], inplace=True)

    # --- Safety checks before touching any files ---
    run_all_safety_checks(dir_list, combined_filemap)

    # Save the move manifest so the operation is recoverable if interrupted
    manifest_path = os.path.join(backup_dir, "combined_filemap.csv")
    combined_filemap.to_csv(manifest_path, index=False)
    print(f"Move manifest saved to: {manifest_path}\n")

    def move_file(src, dest):
        """Move a single file; refuse to overwrite an existing destination."""
        try:
            if os.path.exists(dest):
                return f"✗ Skipped (dest exists): {os.path.basename(dest)}"
            shutil.move(src, dest)
            return f"✓ Moved: {os.path.basename(src)}"
        except Exception as e:
            return f"✗ Failed to move {os.path.basename(src)}: {str(e)}"

    results = Parallel(n_jobs=32, backend="threading")(
        delayed(move_file)(src, dest)
        for src, dest in tqdm(
            zip(combined_filemap["ImagePath"], combined_filemap["OutputPath"]),
            total=len(combined_filemap),
            desc="Moving files",
        )
    )

    # Post-move summary
    failures = [r for r in results if r.startswith("✗")]
    print(
        f"\nDone. {len(results) - len(failures)}/{len(results)} files moved successfully."
    )
    if failures:
        print(f"{len(failures)} failure(s):")
        for f in failures:
            print(f"  {f}")

    return combined_filemap


combined_filemap = combine_all_and_move(dir_list, output_dir)
