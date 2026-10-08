import os
import re

import numpy as np
import polars as pl
from align_toolbox.data_analysis import compute_series_at_time_classified
from align_toolbox.foundation import image_handling
from align_toolbox.foundation.file_handling import read_filemap, write_filemap
from align_toolbox.foundation.utils import find_best_string_match
from align_toolbox.foundation.worm_features import get_features_to_compute_at_molt

# constant definitions
FEATURES_TO_COMPUTE_AT_MOLT = get_features_to_compute_at_molt()
ECDYSIS_COLUMNS = ["HatchTime", "M1", "M2", "M3", "M4"]
MOLT_ENTRY_COLUMNS = ["M1Entry", "M2Entry", "M3Entry", "M4Entry"]
# Events that get a feature-value-at-event column in the GUI. ECDYSIS_COLUMNS
# stays the canonical developmental-events constant for downstream consumers.
VALUE_AT_COLUMNS = ECDYSIS_COLUMNS + MOLT_ENTRY_COLUMNS

KEY_CONVERSION_MAP = {
    "vol": "volume",
    "len": "length",
    "strClass": "qc",
    "ecdys": "ecdysis",
}


def fix_experiment_time(filemap):
    if "ExperimentTime" not in filemap.columns:
        filemap = filemap.with_columns(pl.lit(np.nan).alias("ExperimentTime"))
    filemap = filemap.with_columns(pl.col("ExperimentTime").cast(pl.Float64))

    if filemap.select(pl.col("ExperimentTime")).drop_nulls().is_empty():
        filemap = filemap.with_columns(pl.lit(np.nan).alias("ExperimentTime"))
    return filemap


def get_backup_path(filemap_folder, filemap_name, filemap_extension):
    # check if the filemap is already annotated
    match = re.search(r"annotated_v(/d+)", filemap_name)
    if not match:
        iteration = 1
    else:
        iteration = int(match.group(1))

    filemap_save_path = f"{filemap_name}_v{iteration}{filemap_extension}"
    while os.path.exists(os.path.join(filemap_folder, filemap_save_path)):
        iteration += 1
        filemap_save_path = f"{filemap_name}_v{iteration}{filemap_extension}"

    filemap_save_path = os.path.join(filemap_folder, filemap_save_path)
    return filemap_save_path


def _drop_join_artifact_columns(filemap):
    """Drop polars join-artifact columns left over from an earlier bug.

    A `<base>_right` column paired with an existing `<base>` column is the
    signature of an accidental self-join collision (the default `_right`
    suffix). These carry no meaningful data and must never be persisted; older
    annotated filemaps may already contain them, so we heal them on open.
    """
    columns = set(filemap.columns)
    artifacts = [
        col
        for col in filemap.columns
        if col.endswith("_right") and col[: -len("_right")] in columns
    ]
    if artifacts:
        filemap = filemap.drop(artifacts)
    return filemap


def open_filemap(filemap_path, open_annotated=True, lazy_loading=False):
    filemap_folder = os.path.dirname(filemap_path)
    filemap_name, filemap_extension = os.path.splitext(os.path.basename(filemap_path))
    annotated_name = f"{filemap_name}_annotated{filemap_extension}"
    annotated_path = os.path.join(filemap_folder, annotated_name)

    # If we want to open the annotated version and it's not already annotated
    if (
        open_annotated
        and os.path.exists(annotated_path)
        and ("annotated" not in filemap_name)
    ):
        print(f"Annotated filemap already exists at {annotated_path}")
        print("Opening the existing annotated filemap instead ...")
        filemap_path = annotated_path
        filemap_save_path = annotated_path
        filemap_name = os.path.basename(filemap_path).split(".")[0]
    elif "annotated" not in filemap_name:
        filemap_save_path = annotated_path
    elif "annotated" in filemap_name:
        filemap_save_path = filemap_path

    filemap = read_filemap(filemap_path, lazy_loading=lazy_loading)
    # replace worm_type columns with qc columns
    for col in filemap.columns:
        if "worm_type" in col:
            try:
                filemap = filemap.rename({col: col.replace("worm_type", "qc")})
            except pl.exceptions.DuplicateError:
                print(
                    f"Duplicate column encountered when renaming {col}, dropping it instead."
                )
                filemap = filemap.drop(col)
    # Heal join-artifact `_right` columns leaked by an earlier save-collision bug.
    filemap = _drop_join_artifact_columns(filemap)
    # Backup the filemap
    backup_path = get_backup_path(filemap_folder, filemap_name, filemap_extension)
    write_filemap(filemap, backup_path)

    return filemap, filemap_save_path


def check_use_experiment_time(filemap):
    """
    Check if the filemap contains the 'ExperimentTime' column and if this column contains valid data.
    """
    if "ExperimentTime" in filemap.columns:
        experiment_time = (
            filemap.select(pl.col("ExperimentTime")).to_numpy().squeeze().astype(float)
        )
        if np.any(np.isfinite(experiment_time)):
            return True
        else:
            return False
    else:
        return False


def infer_n_channels(filemap, raw_column="raw"):
    first_image_path = filemap.select(pl.col(raw_column)).to_numpy().squeeze()[0]
    first_image = image_handling.read_tiff_file(first_image_path)

    if first_image.ndim == 3:
        n_channels = first_image.shape[0]
    elif first_image.ndim == 4:
        n_channels = first_image.shape[1]
    elif first_image.ndim == 2:
        n_channels = 1
    else:
        raise ValueError("Unknown number of channels")

    return n_channels


def populate_column_choices(filemap):
    usual_columns = (
        [
            "Time",
            "ExperimentTime",
            "Point",
        ]
        + ECDYSIS_COLUMNS
        + MOLT_ENTRY_COLUMNS
        + [
            "Arrest",
            "Ignore",
            "Death",
            "Dead",
        ]
    )

    raw_columns = [column for column in filemap.columns if "raw" in column]
    usual_columns.extend(raw_columns)
    raw_column = raw_columns[0]
    usual_columns.extend([column for column in filemap.columns if "qc" in column])

    for feature in FEATURES_TO_COMPUTE_AT_MOLT:
        usual_columns.extend(
            [column for column in filemap.columns if feature in column]
        )

    usual_columns.extend([column for column in filemap.columns if "analysis" in column])

    feature_columns = []
    for feature in FEATURES_TO_COMPUTE_AT_MOLT:
        feature_columns.extend(
            [
                column
                for column in filemap.columns
                if feature in column and "_at_" not in column
            ]
        )

    custom_columns = [
        column for column in filemap.columns if column not in usual_columns
    ]

    # add None to the list of custom columns
    custom_columns_choices = ["None"] + custom_columns

    qc_columns = [column for column in filemap.columns if "qc" in column]
    if len(qc_columns) == 0:
        print("No qc column found in the filemap, creating a placeholder.")
        qc_column = "placeholder_qc"
        filemap = filemap.with_columns(pl.lit("worm").alias(qc_column))

    try:
        default_plotted_column = [
            column
            for column in filemap.columns
            if "volume" in column and "_at_" not in column
        ][0]
    except IndexError:
        try:
            default_plotted_column = feature_columns[0]
        except IndexError:
            default_plotted_column = "placeholder_feature"
            filemap = filemap.with_columns(pl.lit(1.0).alias(default_plotted_column))
            feature_columns.append(default_plotted_column)
            print("No feature column found in the filemap, creating a placeholder.")

    segmentation_columns = [
        column for column in filemap.columns if "seg" in column and "str" not in column
    ]

    for feature in FEATURES_TO_COMPUTE_AT_MOLT:
        segmentation_columns = [
            column for column in segmentation_columns if feature not in column
        ]

    overlay_segmentation_choices = ["None"] + segmentation_columns

    return (
        filemap,
        raw_column,
        feature_columns,
        custom_columns_choices,
        default_plotted_column,
        overlay_segmentation_choices,
    )


def separate_column_by_point(filemap, column):
    points = (
        filemap.select(pl.col("Point").unique(maintain_order=True).sort())
        .to_numpy()
        .flatten()
    )

    filemap_points = filemap.select(pl.col("Point"), pl.col(column))
    point_dataframes = filemap_points.partition_by("Point", maintain_order=True)

    sample = point_dataframes[0].select(pl.col(column)).head(1).item()
    is_string = isinstance(sample, str) or (
        hasattr(sample, "dtype") and np.issubdtype(sample.dtype, np.str_)
    )

    max_height = max(point_df.height for point_df in point_dataframes)
    if is_string:
        result = np.full((len(points), max_height), "error", dtype=object)
    else:
        result = np.full((len(points), max_height), np.nan)

    for i, point_df in enumerate(point_dataframes):
        point_column = point_df.select(pl.col(column)).to_numpy().squeeze()
        result[i, : len(point_column)] = point_column
    return result


def get_time_and_ecdysis(filemap):
    ecdysis_df = filemap.select(pl.col(VALUE_AT_COLUMNS + ["Point"]))
    ecdysis_time = (
        (
            ecdysis_df.group_by("Point", maintain_order=True)
            .agg(pl.col(VALUE_AT_COLUMNS).first())
            .select(pl.col(VALUE_AT_COLUMNS))
        )
        .to_numpy()
        .squeeze()
    )

    time = separate_column_by_point(filemap, "Time").astype(float)

    experiment_time = separate_column_by_point(filemap, "ExperimentTime").astype(float)

    if np.ndim(ecdysis_time) < 2:
        ecdysis_time = ecdysis_time[np.newaxis, :]
    if np.ndim(time) < 2:
        time = time[np.newaxis, :]
    if np.ndim(experiment_time) < 2:
        experiment_time = experiment_time[np.newaxis, :]

    # convert experiment_time to hours
    experiment_time = experiment_time / 3600

    ecdysis_index = []

    for i in range(len(ecdysis_time)):
        time_of_point = time[i]
        ecdysis_of_point = ecdysis_time[i]

        ecdysis_index.append(
            [
                (
                    float(np.where(time_of_point == ecdysis)[0][0])
                    if ecdysis in time_of_point
                    else np.nan
                )
                for ecdysis in ecdysis_of_point
            ]
        )

    ecdysis_index = np.array(ecdysis_index)

    return time, experiment_time, ecdysis_index


def build_single_values_df(filemap):
    columns = filemap.columns

    for ecdys in VALUE_AT_COLUMNS:
        if ecdys not in columns:
            filemap = filemap.with_columns(pl.lit(np.nan).alias(ecdys))

    columns_to_keep = ["Point"]
    columns_to_keep.extend([col for col in columns if "_at_" in col])
    columns_to_keep.extend(VALUE_AT_COLUMNS)

    single_values_df = filemap.select(pl.col(columns_to_keep))

    columns_to_keep.remove("Point")

    print(single_values_df.columns)
    single_values_df = single_values_df.group_by("Point", maintain_order=True).agg(
        pl.col(columns_to_keep).first()
    )

    single_values_df = single_values_df.with_columns(
        [pl.col(col).cast(pl.Float64) for col in columns_to_keep]
    )

    return single_values_df


def _ignored_frames_expr(filemap):
    if "Ignore" not in filemap.columns:
        return pl.lit(False)
    return pl.col("Ignore").cast(pl.Boolean).fill_null(False)


def get_ignored_frames(point_filemap):
    """Boolean mask of the frames flagged with Ignore."""
    if "Ignore" not in point_filemap.columns:
        return np.zeros(point_filemap.height, dtype=bool)
    return (
        point_filemap.select(_ignored_frames_expr(point_filemap))
        .to_series()
        .to_numpy()
        .astype(bool)
    )


def process_feature_at_molt_columns(
    filemap, feature_columns, recompute_features_at_molt=False
):
    for ecdys in VALUE_AT_COLUMNS:
        if ecdys not in filemap.columns:
            filemap = filemap.with_columns(pl.lit(np.nan).alias(ecdys))

    if "ExperimentTime" not in filemap.columns:
        filemap = fix_experiment_time(filemap)

    filemap = filemap.with_columns(
        [pl.col(feature_column).cast(pl.Float64) for feature_column in feature_columns]
        + [
            pl.lit(np.nan).alias(f"{feature_column}_at_{ecdys}")
            for feature_column in feature_columns
            for ecdys in VALUE_AT_COLUMNS
            if f"{feature_column}_at_{ecdys}" not in filemap.columns
        ]
    )

    # values at molt are computed without the ignored frames, like the
    # plotting structure of align_toolbox does; fully ignored points keep
    # their stored values
    kept_filemap = filemap.filter(~_ignored_frames_expr(filemap))
    if kept_filemap.height == 0:
        return filemap

    (
        time,
        experiment_time,
        ecdysis_index,
    ) = get_time_and_ecdysis(kept_filemap)

    unique_points = (
        kept_filemap.select(pl.col("Point"))
        .unique(maintain_order=True)
        .to_numpy()
        .squeeze()
    )

    if unique_points.ndim == 0:
        unique_points = np.array([unique_points])

    qc_columns = [column for column in filemap.columns if "qc" in column]

    for feature_column in feature_columns:
        series = separate_column_by_point(kept_filemap, feature_column)
        if len(qc_columns) == 0:
            # No qc column at all: treat every timepoint as a valid worm, matching
            # the placeholder_qc default in populate_column_choices.
            qcs = np.full(series.shape, "worm", dtype=object)
        elif len(qc_columns) == 1:
            qcs = separate_column_by_point(kept_filemap, qc_columns[0])
        else:
            qc_column = find_best_string_match(feature_column, qc_columns)
            qcs = separate_column_by_point(kept_filemap, qc_column)
        feature_at_ecdysis_columns = [
            f"{feature_column}_at_{ecdys}" for ecdys in VALUE_AT_COLUMNS
        ]

        series_at_ecdysis = _get_values_at_molt(kept_filemap, feature_column)

        new_series_at_ecdysis = _compute_series_at_molt(
            series,
            series_at_ecdysis,
            qcs,
            ecdysis_index,
            experiment_time,
            time,
            recompute_values_at_molt=recompute_features_at_molt,
        )

        # check if the new series is different from the old one
        if not np.allclose(series_at_ecdysis, new_series_at_ecdysis, equal_nan=True):
            series_at_ecdysis = new_series_at_ecdysis

        else:
            print(f"{feature_column} at ecdysis is already computed, skipping ...")
            continue

        # For each column, create an expression that handles all points
        updated_df = pl.DataFrame(
            {
                "Point": unique_points,
                **{
                    feature_at_ecdysis_columns[j]: [
                        series_at_ecdysis[i][j] for i in range(len(unique_points))
                    ]
                    for j in range(len(feature_at_ecdysis_columns))
                },
            }
        ).with_columns(pl.col("Point").cast(filemap.schema["Point"]))

        filemap = filemap.join(
            updated_df, on="Point", how="left", suffix="_new", maintain_order="left"
        )
        filemap = filemap.with_columns(
            pl.coalesce(pl.col(f"{column}_new"), pl.col(column)).alias(column)
            for column in feature_at_ecdysis_columns
        ).drop([f"{column}_new" for column in feature_at_ecdysis_columns])

    return filemap


def merge_imported_annotations(
    filemap, imported_df, feature_columns, annotation_columns
):
    """Overlay the annotation columns of an imported filemap onto the GUI's
    filemap and recompute every value at molt from the GUI's own features, qc
    and time. Only the Points present in the import are returned; rows are
    matched on (Point, Time), never by position."""
    annotation_columns = [
        c
        for c in dict.fromkeys(annotation_columns)
        if c in imported_df.columns and c not in ("Point", "Time")
    ]

    imported_annotations = imported_df.select(
        pl.col("Point").cast(filemap.schema["Point"]),
        pl.col("Time").cast(filemap.schema["Time"]),
        *[pl.col(c) for c in annotation_columns],
    )
    imported_points = imported_annotations.select("Point").unique()

    # drop the GUI's values at molt so events missing from the import end up
    # NaN instead of keeping values computed for the GUI's previous events
    stale_columns = annotation_columns + [
        f"{feature_column}_at_{ecdys}"
        for feature_column in feature_columns
        for ecdys in VALUE_AT_COLUMNS
    ]
    merged = (
        filemap.join(imported_points, on="Point", how="semi")
        .drop([c for c in stale_columns if c in filemap.columns])
        .join(imported_annotations, on=["Point", "Time"], how="left")
    )

    return process_feature_at_molt_columns(
        merged, feature_columns, recompute_features_at_molt=True
    )


def clear_values_at_missing_events(filemap):
    """Set every `<feature>_at_<event>` value to NaN on the rows where the
    event itself has no time, so no stale value outlives its event."""
    clear_exprs = []
    for ecdys_event in VALUE_AT_COLUMNS:
        if ecdys_event not in filemap.columns:
            continue
        event_time = pl.col(ecdys_event).cast(pl.Float64)
        event_missing = event_time.is_null() | event_time.is_nan()
        clear_exprs.extend(
            pl.when(event_missing)
            .then(pl.lit(np.nan))
            .otherwise(pl.col(column).cast(pl.Float64))
            .alias(column)
            for column in _get_value_at_event_columns(filemap.columns, ecdys_event)
        )
    return filemap.with_columns(clear_exprs)


def _get_values_at_molt(filemap, column):
    columns_at_ecdysis = [f"{column}_at_{e}" for e in VALUE_AT_COLUMNS]

    column_list = ["Point"] + columns_at_ecdysis
    filemap = filemap.select(pl.col(column_list))

    values_at_ecdysis = (
        (
            filemap.group_by("Point", maintain_order=True)
            .agg(pl.col(columns_at_ecdysis).first())
            .drop("Point")
            .cast(pl.Float64)
        )
        .to_numpy()
        .squeeze()
    )

    return values_at_ecdysis


def _get_value_at_event_columns(columns, ecdys_event):
    # endswith, not substring: "_at_M1" must not match "_at_M1Entry"
    return [column for column in columns if column.endswith(f"_at_{ecdys_event}")]


def update_molt_and_ecdysis_columns(
    point_filemap,
    single_values_df,
    ecdys_event,
    new_time,
    new_time_index,
    experiment_time=True,
):
    single_values_df = single_values_df.with_columns(
        pl.lit(new_time).alias(ecdys_event)
    )

    if experiment_time:
        time = (
            point_filemap.select(pl.col("ExperimentTime")).to_numpy().squeeze() / 3600
        ).astype(float)
    else:
        time = point_filemap.select(pl.col("Time")).to_numpy().squeeze().astype(float)

    value_at_ecdys_columns = _get_value_at_event_columns(
        point_filemap.columns, ecdys_event
    )

    value_columns = [
        re.sub(r"_at_.*$", "", column) for column in value_at_ecdys_columns
    ]

    qc_columns = [col for col in point_filemap.columns if "qc" in col]

    # the ignored frames are left out of the smoothing, and an event on an
    # ignored frame has no value, like in the plotting structure of align_toolbox
    kept_frames = ~get_ignored_frames(point_filemap)
    has_value = (
        not np.isnan(new_time)
        and not np.isnan(new_time_index)
        and kept_frames[int(new_time_index)]
    )

    for value_column, value_at_ecdys_column in zip(
        value_columns, value_at_ecdys_columns
    ):
        if len(qc_columns) == 1:
            qc_values = point_filemap.select(pl.col(qc_columns[0])).to_numpy().squeeze()
        else:
            qc_column = find_best_string_match(value_column, qc_columns)
            qc_values = point_filemap.select(pl.col(qc_column)).to_numpy().squeeze()
        if not has_value:
            new_value_at_ecdys = np.nan
        else:
            series = (
                point_filemap.select(pl.col(value_column)).to_numpy().squeeze().copy()
            )
            new_value_at_ecdys = compute_series_at_time_classified(
                series[kept_frames],
                time[int(new_time_index)],
                time[kept_frames],
                qc_values[kept_frames],
            )

        single_values_df = single_values_df.with_columns(
            pl.lit(new_value_at_ecdys).alias(value_at_ecdys_column)
        )

    return single_values_df


def recompute_values_at_molt_of_point(
    point_filemap, single_values_df, experiment_time=True
):
    """Recompute every `<feature>_at_<event>` value of a point from its
    currently annotated event times, overwriting any pegged values."""
    times = point_filemap.select(pl.col("Time")).to_numpy().ravel().astype(float)

    for ecdys_event in VALUE_AT_COLUMNS:
        if ecdys_event not in single_values_df.columns:
            continue
        event_time = single_values_df.select(pl.col(ecdys_event)).to_numpy().ravel()[0]
        event_time = np.nan if event_time is None else float(event_time)

        matching_indexes = np.where(times == event_time)[0]
        event_time_index = (
            float(matching_indexes[0]) if len(matching_indexes) > 0 else np.nan
        )

        single_values_df = update_molt_and_ecdysis_columns(
            point_filemap,
            single_values_df,
            ecdys_event,
            event_time,
            event_time_index,
            experiment_time=experiment_time,
        )

    return single_values_df


def correct_ecdysis_columns(point_filemap, single_values_df, ecdys_event, time_index):
    value_at_ecdys_columns = _get_value_at_event_columns(
        point_filemap.columns, ecdys_event
    )

    value_columns = [
        re.sub(r"_at_.*$", "", column) for column in value_at_ecdys_columns
    ]

    for value_column, value_at_ecdys_column in zip(
        value_columns, value_at_ecdys_columns
    ):
        new_value_at_ecdys = (
            point_filemap.select(pl.col(value_column))
            .to_numpy()
            .squeeze()
            .copy()[int(time_index)]
        )
        single_values_df = single_values_df.with_columns(
            pl.lit(new_value_at_ecdys).alias(value_at_ecdys_column)
        )

    return single_values_df


def _compute_series_at_molt(
    series,
    series_at_ecdysis,
    qcs,
    ecdysis_index,
    experiment_time,
    time,
    recompute_values_at_molt=False,
):
    if series_at_ecdysis.ndim < 2:
        series_at_ecdysis = series_at_ecdysis[np.newaxis, :]

    new_series_at_ecdysis = series_at_ecdysis.copy()

    nan_indexes_values_mask = np.isnan(series_at_ecdysis)

    if (~np.isnan(experiment_time)).any():
        time = experiment_time
    else:
        time = time

    ecdysis = ecdysis_index

    non_nan_indexes_ecdysis_mask = np.invert(np.isnan(ecdysis))

    if recompute_values_at_molt:
        values_to_recompute_mask = non_nan_indexes_ecdysis_mask
        # an event whose frame is missing (e.g. ignored) has no value
        new_series_at_ecdysis[~non_nan_indexes_ecdysis_mask] = np.nan
    else:
        values_to_recompute_mask = (
            nan_indexes_values_mask & non_nan_indexes_ecdysis_mask
        )

    for i in range(len(values_to_recompute_mask)):
        mask = values_to_recompute_mask[i]
        idx_values_to_recompute = np.where(mask)[0]

        if len(idx_values_to_recompute) == 0:
            continue

        ecdys = ecdysis[i][idx_values_to_recompute].astype(int)

        recomputed_values = compute_series_at_time_classified(
            series[i],
            time[i][ecdys],
            time[i],
            qcs[i],
        )
        new_series_at_ecdysis[i][idx_values_to_recompute] = recomputed_values

    return new_series_at_ecdysis


def set_marker_shape(
    times_of_point,
    selected_time_index,
    qcs,
    hatch_time,
    m1,
    m2,
    m3,
    m4,
    custom_annotations: list = [],
    dark_mode: bool = False,
    m1_entry=np.nan,
    m2_entry=np.nan,
    m3_entry=np.nan,
    m4_entry=np.nan,
):
    symbols = []
    for qc in qcs:
        if qc == "egg":
            symbol = "square-open"
        elif qc == "worm":
            symbol = "circle-open"
        elif qc != "":
            symbol = "triangle-up-open"
        else:
            symbol = ""
        symbols.append(symbol)

    sizes = [4] * len(symbols)
    default_color = "#e0e0e0" if dark_mode else "black"
    colors = [default_color] * len(symbols)

    for custom_annotation in custom_annotations:
        if np.isfinite(custom_annotation):
            symbols[int(custom_annotation)] = "circle"
            sizes[int(custom_annotation)] = 12
            colors[int(custom_annotation)] = "pink"

    if np.isfinite(hatch_time):
        try:
            hatch_index = np.where(times_of_point == hatch_time)[0][0]
            symbols[hatch_index] = "square"
            sizes[hatch_index] = 8
            colors[hatch_index] = "red"
        except IndexError:
            print(f"Hatch time {hatch_time} not in list of times")

    if np.isfinite(m1) and m1 in times_of_point:
        try:
            m1_index = np.where(times_of_point == m1)[0][0]
            symbols[m1_index] = "circle"
            sizes[m1_index] = 8
            colors[m1_index] = "orange"
        except IndexError:
            print(f"M1 {m1} not in list of times")

    if np.isfinite(m2):
        try:
            m2_index = np.where(times_of_point == m2)[0][0]
            symbols[m2_index] = "circle"
            sizes[m2_index] = 8
            colors[m2_index] = "yellow"
        except IndexError:
            print(f"M2 {m2} not in list of times")

    if np.isfinite(m3):
        try:
            m3_index = np.where(times_of_point == m3)[0][0]
            symbols[m3_index] = "circle"
            sizes[m3_index] = 8
            colors[m3_index] = "green"
        except IndexError:
            print(f"M3 {m3} not in list of times")

    if np.isfinite(m4):
        try:
            m4_index = np.where(times_of_point == m4)[0][0]
            symbols[m4_index] = "circle"
            sizes[m4_index] = 8
            colors[m4_index] = "blue"
        except IndexError:
            print(f"M4 {m4} not in list of times")

    entry_specs = [
        (m1_entry, "orange"),
        (m2_entry, "yellow"),
        (m3_entry, "green"),
        (m4_entry, "blue"),
    ]
    for entry_time, color in entry_specs:
        if np.isfinite(entry_time) and entry_time in times_of_point:
            try:
                entry_index = np.where(times_of_point == entry_time)[0][0]
                symbols[entry_index] = "diamond"
                sizes[entry_index] = 8
                colors[entry_index] = color
            except IndexError:
                print(f"Molt entry {entry_time} not in list of times")

    widths = [1] * len(symbols)
    widths[int(selected_time_index)] = 4

    # find the index of all empty symbols
    symbols, sizes, colors, widths = zip(
        *[
            (symbol, size, color, width)
            for symbol, size, color, width in zip(symbols, sizes, colors, widths)
            if symbol != ""
        ]
    )

    markers = dict(symbol=symbols, size=sizes, color=colors, line=dict(width=widths))
    return markers


def get_points_for_value_at_molts(
    hatch,
    m1,
    m2,
    m3,
    m4,
    value_at_hatch,
    value_at_m1,
    value_at_m2,
    value_at_m3,
    value_at_m4,
    m1_entry=np.nan,
    m2_entry=np.nan,
    m3_entry=np.nan,
    m4_entry=np.nan,
    value_at_m1_entry=np.nan,
    value_at_m2_entry=np.nan,
    value_at_m3_entry=np.nan,
    value_at_m4_entry=np.nan,
):
    ecdys_list = [hatch, m1, m2, m3, m4, m1_entry, m2_entry, m3_entry, m4_entry]
    value_at_ecdys_list = [
        value_at_hatch,
        value_at_m1,
        value_at_m2,
        value_at_m3,
        value_at_m4,
        value_at_m1_entry,
        value_at_m2_entry,
        value_at_m3_entry,
        value_at_m4_entry,
    ]
    symbols = ["cross"] * 5 + ["x"] * 4
    colors = [
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "orange",
        "yellow",
        "green",
        "blue",
    ]
    sizes = [8] * 9
    widths = [4] * 9

    # Use numpy to handle NaN values efficiently
    ecdys_array = np.array(ecdys_list)
    value_at_ecdys_array = np.array(value_at_ecdys_list)
    valid_mask = np.isfinite(ecdys_array) & np.isfinite(value_at_ecdys_array)

    # Filter arrays using the mask
    ecdys_filtered = ecdys_array[valid_mask]
    value_at_ecdys_filtered = value_at_ecdys_array[valid_mask]
    symbols_filtered = np.array(symbols)[valid_mask]
    colors_filtered = np.array(colors)[valid_mask]
    sizes_filtered = np.array(sizes)[valid_mask]
    widths_filtered = np.array(widths)[valid_mask]

    return (
        ecdys_filtered,
        value_at_ecdys_filtered,
        symbols_filtered,
        colors_filtered,
        sizes_filtered,
        widths_filtered,
    )


def get_molt_interval_bands(entry_times, exit_times, colors, opacity=0.15):
    """Build Plotly rect shapes spanning each molt's entry->exit interval.

    A band is produced only for molts where both endpoints are finite numbers.
    """

    def _finite(val):
        try:
            return np.isfinite(float(val))
        except (TypeError, ValueError):
            return False

    bands = []
    for entry, exit_time, color in zip(entry_times, exit_times, colors):
        if _finite(entry) and _finite(exit_time):
            x0, x1 = sorted([float(entry), float(exit_time)])
            bands.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=x0,
                    x1=x1,
                    y0=0,
                    y1=1,
                    fillcolor=color,
                    opacity=opacity,
                    layer="below",
                    line_width=0,
                )
            )
    return bands
