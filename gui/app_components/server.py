import asyncio
import traceback

import numpy as np
import plotly.graph_objs as go
import polars as pl
import scipy.io as sio
from align_toolbox.foundation import image_handling
from align_toolbox.foundation.file_handling import read_filemap, write_filemap
from align_toolbox.foundation.utils import find_best_string_match
from app_components.backend import (
    ECDYSIS_COLUMNS,
    KEY_CONVERSION_MAP,
    MOLT_ENTRY_COLUMNS,
    VALUE_AT_COLUMNS,
    build_single_values_df,
    check_use_experiment_time,
    get_molt_interval_bands,
    get_points_for_value_at_molts,
    populate_column_choices,
    process_feature_at_molt_columns,
    set_marker_shape,
)
from app_components.image_cache import (
    BackgroundLoader,
    PointImageCache,
    ProgressTracker,
    array_to_data_url,
    compose_display_image,
    downsample,
    extract_channel,
    prepare_channel,
)
from app_components.ui_components import (
    molt_annotation_buttons_server,
    time_point_navigator_server,
)
from polars.exceptions import ColumnNotFoundError
from shiny import reactive, render, ui
from shinywidgets import render_widget


def main_server(
    input,
    output,
    session,
    filemap=None,
    filemap_save_path=None,
    points=None,
    times=None,
    raw_column="raw",
    feature_columns=None,
    custom_columns_choices=None,
    default_plotted_column=None,
    n_channels=None,
):
    use_experiment_time = check_use_experiment_time(filemap)
    point_filemaps = filemap.partition_by("Point", maintain_order=True)

    _image_cache = PointImageCache()
    _progress_tracker = ProgressTracker()
    _current_loader: list = [None]

    def _start_loading(point_index: int) -> None:
        if _current_loader[0] is not None:
            _current_loader[0].cancel()
        paths = point_filemaps[point_index].select(raw_column).to_numpy().squeeze()
        if paths.ndim == 0:
            value = paths.item()
            paths = [value] if value is not None else []
        else:
            paths = [p for p in paths.tolist() if p is not None]
        _image_cache.reset(point=point_index)
        _progress_tracker.reset(total=len(paths))
        _current_loader[0] = BackgroundLoader(
            point=point_index,
            image_paths=paths,
            n_channels=n_channels,
            cache=_image_cache,
            progress_tracker=_progress_tracker,
        )

    work_df_columns = [
        "Time",
        "Point",
        "Death",
        "Ignore",
        "Arrest",
    ] + custom_columns_choices

    columns_to_get = [c for c in work_df_columns if c in filemap.columns]
    # Remove duplicates while preserving order
    seen = set()
    columns_to_get_unique = []
    for c in columns_to_get:
        if c not in seen:
            columns_to_get_unique.append(c)
            seen.add(c)

    print(f"Columns to get: {columns_to_get_unique}")
    work_df = filemap.select(pl.col(columns_to_get_unique))
    work_df = reactive.Value(work_df)

    single_values_df = build_single_values_df(filemap)

    single_values_of_points = single_values_df.partition_by(
        "Point", maintain_order=True
    )

    single_values_of_points = reactive.Value(single_values_of_points)

    custom_columns_list = reactive.Value(custom_columns_choices)

    hatch, m1, m2, m3, m4 = (reactive.Value("") for _ in range(5))
    value_at_hatch, value_at_m1, value_at_m2, value_at_m3, value_at_m4 = (
        reactive.Value("") for _ in range(5)
    )

    m1_entry, m2_entry, m3_entry, m4_entry = (reactive.Value("") for _ in range(4))
    (
        value_at_m1_entry,
        value_at_m2_entry,
        value_at_m3_entry,
        value_at_m4_entry,
    ) = (reactive.Value("") for _ in range(4))

    ignore_start = reactive.Value("")
    death = reactive.Value("")
    arrest = reactive.Value("")

    clicked_time = reactive.Value("")

    current_point_filemap = reactive.Value("")
    single_values_of_point = reactive.Value("")

    custom_column_values = reactive.Value([])
    column_to_plot = reactive.Value(default_plotted_column)

    current_point_index, current_point = time_point_navigator_server(
        "point",
        choices=points,
        all_df=single_values_of_points,
        current_df=single_values_of_point,
        save_on_switch=True,
    )
    current_time_index, current_time = time_point_navigator_server(
        "time", choices=times, clicked_value=clicked_time
    )

    [
        molt_annotation_buttons_server(
            molt_name,
            current_point_filemap,
            single_values_of_point,
            column_to_plot,
            current_time,
            current_time_index,
            molt_time,
            value_at_molt,
            molt_name=molt_name,
            use_experiment_time=use_experiment_time,
        )
        for molt_name, molt_time, value_at_molt in zip(
            ECDYSIS_COLUMNS,
            [hatch, m1, m2, m3, m4],
            [value_at_hatch, value_at_m1, value_at_m2, value_at_m3, value_at_m4],
        )
    ]

    [
        molt_annotation_buttons_server(
            molt_name,
            current_point_filemap,
            single_values_of_point,
            column_to_plot,
            current_time,
            current_time_index,
            molt_time,
            value_at_molt,
            molt_name=molt_name,
            use_experiment_time=use_experiment_time,
        )
        for molt_name, molt_time, value_at_molt in zip(
            MOLT_ENTRY_COLUMNS,
            [m1_entry, m2_entry, m3_entry, m4_entry],
            [
                value_at_m1_entry,
                value_at_m2_entry,
                value_at_m3_entry,
                value_at_m4_entry,
            ],
        )
    ]

    # Create non-reactive holders for the latest values
    latest_point = {"value": None}
    latest_single_values = {"value": None}
    latest_single_values_df = {"value": None}
    latest_work_df = {"value": None}

    _fig = go.FigureWidget()
    _fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="data"))  # trace 0: data
    _fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",  # trace 1: molt markers
            hoverinfo="none",
            hoverlabel=None,
            hoveron=None,
        )
    )

    def _on_click(trace, points, selector):
        for x in points.xs:
            clicked_time.set(str(x))

    _fig.data[0].on_click(_on_click)

    @reactive.Effect
    def cache_latest_for_save():
        latest_point["value"] = int(current_point_index())
        latest_single_values["value"] = single_values_of_point()
        latest_single_values_df["value"] = single_values_of_points()
        latest_work_df["value"] = work_df()

    @reactive.Effect
    def update_point_filemap():
        point_index = int(current_point_index())
        current_point_filemap.set(point_filemaps[point_index])
        single_values_of_point.set(single_values_of_points()[point_index])
        _start_loading(point_index)

    @reactive.poll(lambda: _progress_tracker.get(), interval_secs=0.5)
    def _current_progress():
        return _progress_tracker.get()

    @output
    @render.ui
    def preload_progress():
        completed, total = _current_progress()
        if total == 0 or completed >= total:
            return ui.div()
        pct = (completed / total) * 100
        return ui.div(
            ui.p(
                f"Pre-loading: {completed} / {total} frames",
                class_="small text-muted mb-1",
                style="margin: 0;",
            ),
            ui.div(
                ui.div(
                    class_="progress-bar bg-info",
                    style=f"width: {pct:.0f}%",
                    role="progressbar",
                    aria_valuenow=str(int(pct)),
                    aria_valuemin="0",
                    aria_valuemax="100",
                ),
                class_="progress mb-2",
                style="height: 8px;",
            ),
        )

    @reactive.Effect
    def update_column_to_plot():
        column_to_plot.set(input.column_to_plot())

    @reactive.Effect
    @reactive.event(input.import_file)
    def import_annotations():
        file = input.import_file()
        point = int(current_point_index())
        if file is None:
            return

        _, _, _, custom_columns_choices, _, _ = populate_column_choices(filemap)
        datapath = file[0]["datapath"]

        if datapath.endswith(".csv") or datapath.endswith(".parquet"):
            print(f"Importing annotations from {datapath} ...")
            imported_df = read_filemap(datapath)

            current_work_df = work_df()
            for col in ["Death", "Ignore", "Arrest"] + custom_columns_choices:
                if col in imported_df.columns:
                    continue
                if col in current_work_df.columns:
                    value = current_work_df.select(pl.col(col)).to_numpy().squeeze()
                elif col == "Death":
                    value = np.nan
                elif col in ("Ignore", "Arrest"):
                    value = False
                else:
                    value = np.nan
                imported_df = imported_df.with_columns(pl.lit(value).alias(col))

            for col in ["ExperimentTime"] + feature_columns:
                if col not in imported_df.columns and col in filemap.columns:
                    imported_df = imported_df.with_columns(
                        pl.lit(filemap.select(pl.col(col)).to_numpy().squeeze()).alias(
                            col
                        )
                    )

            # Always classify frames using the qc of the filemap currently open in
            # the GUI, never the qc the imported file happens to carry. Drop any
            # imported qc so values-at-molt are computed against the GUI's qc only.
            imported_df = imported_df.drop(
                [col for col in imported_df.columns if "qc" in col]
            )
            for col in [col for col in filemap.columns if "qc" in col]:
                imported_df = imported_df.with_columns(
                    pl.lit(filemap.select(pl.col(col)).to_numpy().squeeze()).alias(col)
                )

            imported_df = process_feature_at_molt_columns(
                imported_df, feature_columns, recompute_features_at_molt=False
            )

            work_cols = list(
                dict.fromkeys(
                    [c for c in columns_to_get_unique if c in imported_df.columns]
                    + [
                        c
                        for c in ["Death", "Ignore", "Arrest"]
                        if c in imported_df.columns
                    ]
                )
            )
            work_cols_to_update = [c for c in work_cols if c not in ("Point", "Time")]
            new_work_df = imported_df.select(pl.col(work_cols))

            import_keys = new_work_df.select(["Point", "Time"]).with_columns(
                pl.lit(True).alias("_in_import")
            )
            tagged = current_work_df.join(
                import_keys, on=["Point", "Time"], how="left"
            ).join(
                new_work_df.select(["Point", "Time"] + work_cols_to_update),
                on=["Point", "Time"],
                how="left",
                suffix="_new",
            )
            update_exprs = [
                pl.when(pl.col("_in_import"))
                .then(pl.col(f"{c}_new"))
                .otherwise(pl.col(c))
                .alias(c)
                for c in work_cols_to_update
                if c in current_work_df.columns
            ]
            drop_cols = ["_in_import"] + [
                f"{c}_new" for c in work_cols_to_update if c in current_work_df.columns
            ]
            work_df.set(tagged.with_columns(update_exprs).drop(drop_cols))

            import_single_values_df = build_single_values_df(imported_df)

        elif datapath.endswith(".mat"):
            matlab_report = sio.loadmat(datapath, chars_as_strings=False)
            new_matlab_report = {}
            for key, value in matlab_report.items():
                if key.startswith("__"):
                    continue
                new_key = KEY_CONVERSION_MAP.get(key)
                if new_key:
                    new_matlab_report[new_key] = value

            try:
                ecdysis = new_matlab_report["ecdysis"]
            except KeyError:
                print("No molts found in the matlab report")
                return

            ecdysis_df = pl.DataFrame(
                {
                    "Point": range(len(ecdysis)),
                    **{
                        col: [
                            float(molts[i]) - 1 if not np.isnan(molts[i]) else np.nan
                            for molts in ecdysis
                        ]
                        for i, col in enumerate(ECDYSIS_COLUMNS)
                    },
                }
            )
            imported_df = process_feature_at_molt_columns(
                filemap.drop(ECDYSIS_COLUMNS).join(ecdysis_df, on="Point", how="left"),
                feature_columns,
                recompute_features_at_molt=True,
            )
            import_single_values_df = build_single_values_df(imported_df)

        else:
            return

        current_combined = pl.concat(single_values_of_points(), how="vertical")
        cols_to_update = [
            c
            for c in import_single_values_df.columns
            if c != "Point" and c in current_combined.columns
        ]

        import_point_keys = import_single_values_df.select("Point").with_columns(
            pl.lit(True).alias("_in_import")
        )
        tagged = current_combined.join(import_point_keys, on="Point", how="left").join(
            import_single_values_df.select(["Point"] + cols_to_update),
            on="Point",
            how="left",
            suffix="_new",
        )
        update_exprs = [
            pl.when(pl.col("_in_import"))
            .then(pl.col(f"{c}_new"))
            .otherwise(pl.col(c))
            .alias(c)
            for c in cols_to_update
        ]
        drop_cols = ["_in_import"] + [f"{c}_new" for c in cols_to_update]
        merged = tagged.with_columns(update_exprs).drop(drop_cols)

        merged_partitions = merged.partition_by("Point", maintain_order=True)
        single_values_of_points.set(merged_partitions)
        single_values_of_point.set(merged_partitions[point])

    @reactive.calc
    def get_images_of_point():
        return current_point_filemap().select(raw_column).to_numpy().squeeze().tolist()

    @reactive.calc
    def get_segmentation_of_point():
        if input.segmentation_overlay() == "None":
            return []

        return (
            current_point_filemap()
            .select(input.segmentation_overlay())
            .to_numpy()
            .squeeze()
            .tolist()
        )

    @reactive.calc
    def get_custom_annotations():
        if input.custom_column() == "" or input.custom_column() == "None":
            return []

        return (
            work_df()
            .select(input.custom_column())
            .to_numpy()
            .squeeze()
            .astype(float)
            .tolist()
        )

    def save_filemap(
        filemap,
        filemap_save_path,
        current_index,
        current_single_values_df,
        single_values_df,
        work_df,
    ):
        print(f"Saving filemap to {filemap_save_path} ...")

        # because the list of single values is only updated when the point is changed, change it here to make sure to include the latest changes
        single_values_df[int(current_index)] = current_single_values_df

        single_values_columns = single_values_df[0].columns
        single_values_columns.remove("Point")
        columns_to_drop = [c for c in single_values_columns if c in filemap.columns]
        if len(columns_to_drop) > 0:
            filemap_to_save = filemap.drop(columns_to_drop)
        else:
            filemap_to_save = filemap

        work_df_columns = work_df.columns
        work_df_columns.remove("Point")
        work_df_columns.remove("Time")

        columns_to_drop = [c for c in work_df_columns if c in filemap_to_save.columns]
        if len(columns_to_drop) > 0:
            filemap_to_save = filemap_to_save.drop(columns_to_drop)

        # combine the single_values_df into a big dataframe
        combined_df = pl.concat(single_values_df, how="vertical")

        filemap_to_save = filemap_to_save.join(combined_df, on=["Point"], how="left")

        filemap_to_save = filemap_to_save.join(
            work_df, on=["Point", "Time"], how="left"
        )

        write_filemap(filemap_to_save, filemap_save_path)
        print("Filemap saved!")

    @reactive.Effect
    @reactive.event(input.save)
    def save():
        save_filemap(
            filemap=filemap,
            current_index=current_point_index(),
            current_single_values_df=single_values_of_point(),
            filemap_save_path=filemap_save_path,
            single_values_df=single_values_of_points(),
            work_df=work_df(),
        )

    @reactive.Effect
    def get_hatch_and_molts():
        sv = single_values_of_point()
        if not isinstance(sv, pl.DataFrame):
            return
        # Get all molt times
        hatch_and_molts = sv.select(pl.col(ECDYSIS_COLUMNS)).to_numpy().squeeze()

        # Set molt times using unpacking
        molt_values = [hatch, m1, m2, m3, m4]
        for i, molt_value in enumerate(molt_values):
            molt_value.set(hatch_and_molts[i])

        entry_times = sv.select(pl.col(MOLT_ENTRY_COLUMNS)).to_numpy().squeeze()
        entry_values = [m1_entry, m2_entry, m3_entry, m4_entry]
        for i, entry_value in enumerate(entry_values):
            entry_value.set(entry_times[i])

        # Get values at each molt
        try:
            column = input.column_to_plot()
            molt_feature_values = [
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

            for molt_value, molt_name in zip(molt_feature_values, VALUE_AT_COLUMNS):
                molt_value.set(
                    single_values_of_point()
                    .select(pl.col(f"{column}_at_{molt_name}"))
                    .to_numpy()
                    .squeeze()
                )
        except Exception as e:
            print(f"Exception caught while getting value at hatch and molts: {e}")
            for molt_value in molt_feature_values:
                molt_value.set(np.nan)

    @reactive.Effect
    def set_ignore_start():
        try:
            df = work_df().filter(pl.col("Point") == int(current_point()))
            print(df)
            ignore_values = df.select(pl.col("Ignore")).to_numpy().squeeze()
            time_values = df.select(pl.col("Time")).to_numpy().squeeze()
            # find the first True value
            true_indices = np.where(ignore_values)[0]
            if true_indices.size > 0:
                ignore_start.set(time_values[true_indices[0]])
            else:
                ignore_start.set("")

        except (ColumnNotFoundError, ValueError):
            ignore_start.set("")

        print(f"Ignore start: {ignore_start()}")

    @reactive.Effect
    @reactive.event(input.set_ignore_point)
    def set_ignore_point():
        point = int(current_point())
        df = work_df()

        print(f"Setting ignore for point {point}")
        if "Ignore" not in df.columns:
            df = df.with_columns(pl.lit(False).alias("Ignore"))

        if (
            df.filter(pl.col("Point") == point)
            .select(pl.col("Ignore"))
            .to_numpy()
            .all()
        ):
            work_df.set(
                df.with_columns(
                    pl.when(pl.col("Point") == point)
                    .then(False)
                    .otherwise(pl.col("Ignore"))
                    .alias("Ignore")
                )
            )
        else:
            work_df.set(
                df.with_columns(
                    pl.when(pl.col("Point") == point)
                    .then(True)
                    .otherwise(pl.col("Ignore"))
                    .alias("Ignore")
                )
            )

        print(work_df().filter(pl.col("Point") == point).select(pl.col("Ignore")))

    @reactive.Effect
    @reactive.event(input.set_ignore_after)
    def set_ignore_after():
        point = int(current_point())
        time = int(current_time())
        df = work_df()
        print(f"Setting ignore after for point {point}")
        if "Ignore" not in df.columns:
            df = df.with_columns(pl.lit(False).alias("Ignore"))

        if (
            df.filter((pl.col("Point") == point) & (pl.col("Time") >= time))
            .select(pl.col("Ignore"))
            .to_numpy()
            .all()
        ):
            work_df.set(
                df.with_columns(
                    pl.when((pl.col("Point") == point) & (pl.col("Time") >= time))
                    .then(False)
                    .otherwise(pl.col("Ignore"))
                    .alias("Ignore")
                )
            )
        else:
            work_df.set(
                df.with_columns(
                    pl.when((pl.col("Point") == point) & (pl.col("Time") >= time))
                    .then(True)
                    .otherwise(pl.col("Ignore"))
                    .alias("Ignore")
                )
            )

        print(work_df().filter(pl.col("Point") == point).select(pl.col("Ignore")))

    @reactive.Effect
    def get_death():
        try:
            d = (
                work_df()
                .filter(pl.col("Point") == int(current_point()))
                .select(pl.col("Death"))
                .to_numpy()
                .squeeze()[0]
            )
            if d is None:
                death.set("")
            else:
                death.set(d)
        except ColumnNotFoundError:
            death.set("")

    @reactive.Effect
    @reactive.event(input.set_death)
    def set_death():
        point = int(current_point())
        time = int(current_time())

        df = work_df()

        if "Death" not in df.columns:
            df = df.with_columns(pl.lit(np.nan).alias("Death"))

        point_df = df.filter(pl.col("Point") == point)

        if point_df.select(pl.col("Death")).to_numpy().squeeze()[0] == time:
            df = df.with_columns(
                pl.when(pl.col("Point") == point)
                .then(np.nan)
                .otherwise(pl.col("Death"))
                .alias("Death")
            )

        else:
            df = df.with_columns(
                pl.when(pl.col("Point") == point)
                .then(time)
                .otherwise(pl.col("Death"))
                .alias("Death")
            )

        work_df.set(df)

    @reactive.Effect
    @reactive.event(input.set_arrest)
    def set_arrest():
        point = int(current_point())

        df = work_df()

        if "Arrest" not in df.columns:
            df = df.with_columns(pl.lit(False).alias("Arrest"))

        point_df = df.filter(pl.col("Point") == point)

        if np.all(point_df.select(pl.col("Arrest")).to_numpy().squeeze()):
            df = df.with_columns(
                pl.when(pl.col("Point") == point)
                .then(False)
                .otherwise(pl.col("Arrest"))
                .alias("Arrest")
            )

        else:
            df = df.with_columns(
                pl.when(pl.col("Point") == point)
                .then(True)
                .otherwise(pl.col("Arrest"))
                .alias("Arrest")
            )

        work_df.set(df)

    @reactive.Effect
    def get_arrest():
        try:
            arrest.set(
                work_df()
                .filter(pl.col("Point") == int(current_point()))
                .select(pl.col("Arrest"))
                .to_numpy()
                .squeeze()[0]
            )
        except ColumnNotFoundError:
            arrest.set(False)

    @reactive.Effect
    def reactive_get_custom_column_values():
        print("reactive_get_custom_column_values")
        custom_column = input.custom_column()

        local_work_df = work_df()
        work_df_columns = local_work_df.columns

        if custom_column == "None" or custom_column == "":
            custom_column_values.set([])
            return
        if custom_column not in work_df_columns:
            local_work_df = local_work_df.with_columns(
                pl.lit(np.nan).alias(custom_column)
            )
            work_df.set(local_work_df)
            custom_column_values.set([])
        else:
            custom_column_values.set(
                get_custom_column_values(
                    local_work_df, custom_column, int(current_point())
                )
            )
        print(f"Custom column values: {custom_column_values()}")

    def get_custom_column_values(df, column, point):
        if column == "None" or column == "":
            return []
        if column not in df.columns:
            df = df.with_columns(pl.lit(np.nan).alias(column))
        intermediate_df = df.filter(pl.col("Point") == point)

        # convert the column to index in the time values
        time = intermediate_df.select(pl.col("Time")).to_numpy().squeeze()
        column_values = (
            intermediate_df.select(pl.col(column)).to_numpy().squeeze().astype(float)
        )

        non_nan_column_values = column_values[~np.isnan(column_values)]
        column_values_as_indexes = np.full_like(non_nan_column_values, np.nan)
        for i, value in enumerate(non_nan_column_values):
            index = np.where(time == value)[0]
            if index.size > 0:
                column_values_as_indexes[i] = index[0]
        return column_values_as_indexes

    @reactive.Effect
    @reactive.event(input.custom_annotation)
    def set_custom_annotation():
        try:
            new_column_name = input.new_custom_column()
            custom_column = input.custom_column()

            local_work_df = work_df()
            filemap_columns = filemap.columns
            custom_columns = custom_columns_list()

            if (new_column_name != "" and new_column_name != "None") and (
                new_column_name not in custom_columns
            ):
                custom_columns.append(new_column_name)
                custom_columns_list.set(custom_columns)
                ui.update_selectize(
                    "custom_column",
                    choices=custom_columns_list(),
                    selected=new_column_name,
                )
                ui.update_text("new_custom_column", value="")
                custom_column = new_column_name

            if custom_column == "" or custom_column == "None":
                return

            if custom_column not in filemap_columns:
                local_work_df = local_work_df.with_columns(
                    pl.lit(np.nan).alias(new_column_name)
                )
            elif custom_column not in local_work_df.columns:
                new_column_values = (
                    filemap.select(pl.col(custom_column)).to_numpy().squeeze()
                )
                local_work_df = local_work_df.with_columns(
                    pl.lit(new_column_values).alias(custom_column)
                )

            time = int(current_time())
            point = int(current_point())

            local_work_df = local_work_df.with_columns(
                pl.when((pl.col("Point") == point) & (pl.col("Time") == time))
                .then(float(time))
                .otherwise(pl.col(custom_column))
                .alias(custom_column),
            )
            work_df.set(local_work_df)

        except Exception as e:
            print(f"Exception in set_custom_annotation: {e}")
            traceback.print_exc()

    @reactive.Effect
    @reactive.event(input.reset_custom_annotation)
    def reset_custom_annotation():
        custom_column = input.custom_column()
        if custom_column != "" and custom_column != "None":
            try:
                work_df.set(
                    work_df().with_columns(
                        pl.when(pl.col("Point") == int(current_point()))
                        .then(np.nan)
                        .otherwise(pl.col(custom_column))
                        .alias(custom_column),
                    )
                )
            except Exception as e:
                print(f"Exception in reset_custom_annotation: {e}")

    @output
    @render.ui
    def plot_image():
        idx = current_time_index()
        channel_str = input.channel()
        if channel_str == "None" or channel_str is None:
            return ui.div()
        channel = int(channel_str.split(" ")[-1]) - 1

        images_of_point = get_images_of_point()
        segmentation_of_point = get_segmentation_of_point()

        if not images_of_point or idx >= len(images_of_point):
            return ui.div()

        channel_overlay_str = input.channel_overlay()
        need_overlay = channel_overlay_str not in ("None", None)
        channel_overlay_idx = (
            int(channel_overlay_str.split(" ")[-1]) - 1 if need_overlay else None
        )

        main_arr = _image_cache.get(idx, channel)
        overlay_arr = (
            _image_cache.get(idx, channel_overlay_idx) if need_overlay else None
        )

        if main_arr is None or (need_overlay and overlay_arr is None):
            raw_img = image_handling.read_tiff_file(images_of_point[idx])
            if main_arr is None:
                main_arr = prepare_channel(extract_channel(raw_img, channel))
            if need_overlay and overlay_arr is None:
                overlay_arr = prepare_channel(
                    extract_channel(raw_img, channel_overlay_idx)
                )

        seg_arr = None
        if len(segmentation_of_point) > 0:
            try:
                seg = image_handling.read_tiff_file(segmentation_of_point[idx])
                if seg.ndim == 2:
                    seg_arr = downsample(seg)
                elif seg.ndim == 3:
                    seg_arr = downsample(seg[seg.shape[0] // 2])
                else:
                    seg_arr = downsample(seg.squeeze())
            except Exception:
                seg_arr = None

        rgb = compose_display_image(
            main_arr=main_arr,
            overlay_arr=overlay_arr,
            seg_arr=seg_arr,
        )
        data_url = array_to_data_url(rgb)
        return ui.img(
            src=data_url,
            style=f"height: {input.curve_plot_height()}px; max-width: 100%; object-fit: contain;",
        )

    @output
    @render_widget
    def plot_curve():
        return _fig

    @reactive.Effect
    def _update_plot_container_size():
        h = input.curve_plot_height()
        w = input.curve_plot_width()
        with _fig.batch_update():
            _fig.update_layout(height=h, width=w)
        asyncio.ensure_future(
            session.send_custom_message("resize_plot", {"height": h, "width": w})
        )

    @reactive.Effect
    def _update_plot_theme():
        mode = input.color_mode()
        if mode is None:
            return
        template = "plotly_dark" if mode == "dark" else "plotly"
        with _fig.batch_update():
            _fig.update_layout(template=template)

    def _is_finite_number(val):
        try:
            return np.isfinite(float(val))
        except (TypeError, ValueError):
            return False

    @reactive.Effect
    def _update_curve():
        # --- data trace ---
        col = input.column_to_plot()
        pf = current_point_filemap()
        dark_mode = input.color_mode() == "dark"

        times_of_point = pf.select(pl.col("Time")).to_numpy().squeeze()
        values_of_point = pf.select(pl.col(col)).to_numpy().squeeze()

        # markers
        qc_columns = [c for c in pf.columns if "qc" in c]
        if len(qc_columns) == 1:
            qc_column = qc_columns[0]
        else:
            qc_column = find_best_string_match(col, qc_columns)

        qcs_of_point = pf.select(pl.col(qc_column)).to_numpy().squeeze()
        markers = set_marker_shape(
            times_of_point,
            current_time_index(),
            qcs_of_point,
            hatch(),
            m1(),
            m2(),
            m3(),
            m4(),
            custom_annotations=custom_column_values(),
            dark_mode=dark_mode,
            m1_entry=m1_entry(),
            m2_entry=m2_entry(),
            m3_entry=m3_entry(),
            m4_entry=m4_entry(),
        )

        # molt scatter
        (
            ecdys_x,
            ecdys_y,
            symbols,
            colors,
            sizes,
            widths,
        ) = get_points_for_value_at_molts(
            hatch(),
            m1(),
            m2(),
            m3(),
            m4(),
            value_at_hatch(),
            value_at_m1(),
            value_at_m2(),
            value_at_m3(),
            value_at_m4(),
            m1_entry=m1_entry(),
            m2_entry=m2_entry(),
            m3_entry=m3_entry(),
            m4_entry=m4_entry(),
            value_at_m1_entry=value_at_m1_entry(),
            value_at_m2_entry=value_at_m2_entry(),
            value_at_m3_entry=value_at_m3_entry(),
            value_at_m4_entry=value_at_m4_entry(),
        )

        # shapes (ignore region, death line, arrest overlay)
        shapes = []
        if _is_finite_number(ignore_start()):
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=float(ignore_start()),
                    x1=float(max(times_of_point)) + 1,
                    y0=0,
                    y1=1,
                    fillcolor="gray",
                    opacity=0.5,
                    layer="above",
                    line_width=0,
                )
            )
        if _is_finite_number(death()):
            shapes.append(
                dict(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=float(death()),
                    y0=0,
                    x1=float(death()),
                    y1=1,
                    line=dict(color="#e0e0e0" if dark_mode else "black", width=2),
                )
            )

        arrest_shapes = []
        if arrest() != "" and arrest():
            arrest_shapes.append(
                dict(
                    type="rect",
                    xref="paper",
                    yref="paper",
                    x0=0,
                    x1=1,
                    y0=0,
                    y1=1,
                    fillcolor="red",
                    opacity=0.5,
                    layer="above",
                    line_width=0,
                )
            )

        band_shapes = get_molt_interval_bands(
            [m1_entry(), m2_entry(), m3_entry(), m4_entry()],
            [m1(), m2(), m3(), m4()],
            ["orange", "yellow", "green", "blue"],
        )

        with _fig.batch_update():
            _fig.layout.shapes = []  # clear first
            _fig.data[0].x = times_of_point
            _fig.data[0].y = values_of_point
            _fig.data[0].marker = markers

            _fig.data[1].x = ecdys_x
            _fig.data[1].y = ecdys_y
            _fig.data[1].marker = dict(
                symbol=symbols,
                size=sizes,
                color=colors,
                line=dict(width=widths, color=colors),
            )

        _fig.update_layout(
            xaxis_title="Time",
            yaxis_title=col,
            margin=dict(l=20, r=20, t=50, b=50),
            showlegend=False,
            yaxis_type="log" if input.log_scale() else "linear",
            shapes=shapes + arrest_shapes + band_shapes,
        )

    def save_on_session_end():
        if latest_point["value"] is None:
            print("Session ended before state was captured; skipping save.")
            return
        save_filemap(
            filemap=filemap,
            current_index=latest_point["value"],
            current_single_values_df=latest_single_values["value"],
            filemap_save_path=filemap_save_path,
            single_values_df=latest_single_values_df["value"],
            work_df=latest_work_df["value"],
        )

    def _on_session_end():
        if _current_loader[0] is not None:
            _current_loader[0].cancel()
        save_on_session_end()

    session.on_ended(_on_session_end)
    return save_on_session_end  # return so the SIGINT handler can call it directly
