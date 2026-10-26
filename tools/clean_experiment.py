"""Clean the frames an annotated experiment no longer needs: frames flagged
Ignore in the GUI and, with --max_time_after_m4, frames more than that many time
points after the individual's M4. The frames are picked with the same rule as the
pipeline's rerun mode.

For those frames:
- the raw and analysis images referenced by the annotated filemap are deleted
  (only files inside the experiment's raw or analysis dir, outside its report dir);
- the per-frame report files (morphology, QC, fluorescence, ...) have every value
  but Time and Point blanked (set to null);
- the filemap and annotated filemap have their image path columns and report
  columns blanked. Annotations (Ignore, M4, ...) and ExperimentTime are kept, so
  the script can be run again and reruns still skip the same frames.

Reports without a Time column (e.g. ecdysis) and the GUI's backup filemaps are
left untouched. Rows are blanked, not removed.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import polars as pl
from align_toolbox.foundation.file_handling import read_filemap, write_filemap

sys.path.append(str(Path(__file__).parent.parent))
from align_pipeline.init_pipeline import SUPPORTED_FILEMAP_FORMATS  # noqa: E402
from align_pipeline.init_pipeline import find_existing_filemap  # noqa: E402
from align_pipeline.utils import get_rerun_skip_rows  # noqa: E402

KEY_COLUMNS = ("Time", "Point")
# Never blanked in the report files.
KEPT_REPORT_COLUMNS = KEY_COLUMNS + ("ExperimentTime",)
FILEMAP_NAMES = ("analysis_filemap", "analysis_filemap_annotated")


def get_args():
    parser = argparse.ArgumentParser(
        description="Delete the raw and analysis images of ignored frames and of "
        "frames too long after M4, and blank them in the filemaps and reports."
    )
    parser.add_argument("experiment_dir", help="Path to the experiment.")
    parser.add_argument(
        "-a",
        "--analysis_dir_name",
        default="analysis",
        help="Name of the analysis folder inside the experiment (default: analysis).",
    )
    parser.add_argument(
        "-m",
        "--max_time_after_m4",
        type=int,
        default=None,
        help="Also clean frames more than this many time points after M4. "
        "If omitted, only frames flagged Ignore are cleaned.",
    )
    parser.add_argument(
        "-r",
        "--raw_dir_name",
        default="raw",
        help="Name of the raw folder inside the experiment (default: raw).",
    )
    parser.add_argument(
        "-n",
        "--dry_run",
        action="store_true",
        help="Report what would be deleted and blanked without changing anything.",
    )
    return parser.parse_args()


def get_report_dirs(experiment_dir, raw_dir_name, analysis_dir_name):
    """Return the report dirs holding the filemaps: analysis/report, or one
    analysis/report/<subdir> per raw subdir in subdirectory mode."""
    raw_dir = os.path.join(experiment_dir, raw_dir_name)
    report_dir = os.path.join(experiment_dir, analysis_dir_name, "report")
    subdirs = sorted(
        f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))
    )
    if not subdirs:
        return [report_dir]
    return [os.path.join(report_dir, subdir) for subdir in subdirs]


def get_report_files(report_dir):
    """Return the per-frame report files of report_dir: every table except the
    filemaps (and their GUI backups) that has Time and Point columns."""
    report_files = []
    for name in sorted(os.listdir(report_dir)):
        path = os.path.join(report_dir, name)
        stem, ext = os.path.splitext(name)
        if not os.path.isfile(path) or ext.lstrip(".") not in SUPPORTED_FILEMAP_FORMATS:
            continue
        if re.fullmatch(r"analysis_filemap(_annotated)?(_v\d+)?", stem):
            continue
        columns = read_filemap(path, lazy_loading=True).collect_schema().names()
        if all(column in columns for column in KEY_COLUMNS):
            report_files.append(path)
    return report_files


def is_inside(path, directory):
    return os.path.commonpath([path, directory]) == directory


class ImagePaths:
    """Resolve filemap values to the image files the script may delete: paths
    inside the raw or analysis dir, outside the report dir."""

    def __init__(self, experiment_dir, raw_dir, analysis_dir, report_root):
        self.experiment_dir = experiment_dir
        self.allowed_dirs = [raw_dir, analysis_dir]
        self.report_root = report_root

    def resolve(self, value):
        """Return the absolute image path value points to, or None."""
        path = os.path.realpath(os.path.join(self.experiment_dir, value))
        if is_inside(path, self.report_root):
            return None
        if any(is_inside(path, directory) for directory in self.allowed_dirs):
            return path
        return None

    def get_columns(self, filemap):
        """Return the String columns of the filemap holding image paths."""
        return [
            column
            for column, dtype in filemap.schema.items()
            if dtype == pl.String
            and column not in KEY_COLUMNS
            and any(
                self.resolve(value) is not None
                for value in filemap.get_column(column).drop_nulls().head(100)
            )
        ]


def select_rows(table, skip_rows):
    skip_rows = skip_rows.cast({column: table.schema[column] for column in KEY_COLUMNS})
    return table.join(skip_rows, on=list(KEY_COLUMNS), how="semi")


def get_files_to_delete(filemap, skip_rows, image_paths):
    """Return the existing image files referenced by the skipped rows."""
    skipped = select_rows(filemap, skip_rows)
    files = set()
    for column in image_paths.get_columns(skipped):
        for value in skipped.get_column(column).drop_nulls().to_list():
            path = image_paths.resolve(value)
            if path is not None and os.path.isfile(path):
                files.add(path)
    return sorted(files)


def blank_rows(table, skip_rows, columns):
    """Set columns to null on the skipped rows of table. Returns the new table
    and the number of rows that had a value blanked."""
    columns = [column for column in columns if column in table.columns]
    if not columns or skip_rows.is_empty():
        return table, 0
    skipped = "__skipped__"
    skip_rows = skip_rows.cast(
        {column: table.schema[column] for column in KEY_COLUMNS}
    ).with_columns(pl.lit(True).alias(skipped))
    table = table.join(
        skip_rows, on=list(KEY_COLUMNS), how="left", maintain_order="left"
    )
    is_skipped = pl.col(skipped).fill_null(False)
    n_blanked = table.filter(
        is_skipped & pl.any_horizontal(pl.col(columns).is_not_null())
    ).height
    table = table.with_columns(
        pl.when(is_skipped).then(None).otherwise(pl.col(column)).alias(column)
        for column in columns
    ).drop(skipped)
    return table, n_blanked


def clean_tables(report_dir, skip_rows, image_paths, dry_run):
    """Blank the skipped rows in the report files and filemaps of report_dir."""
    prefix = "[dry run] would blank" if dry_run else "Blanked"
    report_columns = set()
    for path in get_report_files(report_dir):
        table = read_filemap(path)
        columns = [c for c in table.columns if c not in KEPT_REPORT_COLUMNS]
        report_columns.update(columns)
        table, n_blanked = blank_rows(table, skip_rows, columns)
        print(f"  {prefix} {n_blanked} row(s) in {path}")
        if n_blanked and not dry_run:
            write_filemap(table, path)

    for name in FILEMAP_NAMES:
        path = find_existing_filemap(report_dir, name)
        if path is None:
            continue
        table = read_filemap(path)
        columns = image_paths.get_columns(table) + sorted(report_columns)
        table, n_blanked = blank_rows(table, skip_rows, columns)
        print(f"  {prefix} {n_blanked} row(s) in {path}")
        if n_blanked and not dry_run:
            write_filemap(table, path)


def main():
    args = get_args()
    experiment_dir = os.path.realpath(args.experiment_dir)
    raw_dir = os.path.join(experiment_dir, args.raw_dir_name)
    analysis_dir = os.path.join(experiment_dir, args.analysis_dir_name)
    for directory in (raw_dir, analysis_dir):
        if not os.path.isdir(directory):
            sys.exit(f"Error: {directory} does not exist")
    image_paths = ImagePaths(
        experiment_dir, raw_dir, analysis_dir, os.path.join(analysis_dir, "report")
    )

    total_files, total_size = 0, 0
    for report_dir in get_report_dirs(
        experiment_dir, args.raw_dir_name, args.analysis_dir_name
    ):
        filemap_path = find_existing_filemap(report_dir, "analysis_filemap_annotated")
        if filemap_path is None:
            print(f"No annotated filemap in {report_dir}, skipping")
            continue
        filemap = read_filemap(filemap_path)
        skip_rows = get_rerun_skip_rows(filemap, args.max_time_after_m4)
        files = get_files_to_delete(filemap, skip_rows, image_paths)
        size = sum(os.path.getsize(f) for f in files)
        print(
            f"{filemap_path}: {len(skip_rows)} frame(s) to clean, "
            f"{len(files)} file(s) to delete ({size / 1e9:.2f} GB)"
        )

        # Files go first: if blanking fails midway, running the script again
        # still finds the frames (annotations are kept) and finishes the job.
        for f in files:
            if args.dry_run:
                print(f"  [dry run] would delete {f}")
            else:
                os.remove(f)
        clean_tables(report_dir, skip_rows, image_paths, args.dry_run)
        total_files += len(files)
        total_size += size

    verb = "[dry run] would delete" if args.dry_run else "Deleted"
    print(f"{verb} {total_files} file(s), {total_size / 1e9:.2f} GB in total")


if __name__ == "__main__":
    main()
