from pathlib import Path

import polars as pl
from app_components.backend import (
    ECDYSIS_COLUMNS,
    MOLT_ENTRY_COLUMNS,
    fix_experiment_time,
    infer_n_channels,
    populate_column_choices,
    process_feature_at_molt_columns,
)
from app_components.ui_components import molt_annotation_buttons, time_point_navigator
from shiny import ui
from shinywidgets import output_widget


def create_molt_annotator(ecdysis_list_id, entry_list_id, custom_columns_choices):
    molt_columns = []
    for molt in ecdysis_list_id:
        entry = f"{molt}Entry"
        if entry in entry_list_id:
            molt_columns.append(
                ui.column(
                    2,
                    molt_annotation_buttons(molt, molt=molt),
                    ui.hr(),
                    molt_annotation_buttons(entry, molt=entry),
                )
            )
        else:
            molt_columns.append(ui.column(2, molt_annotation_buttons(molt, molt=molt)))

    return ui.column(
        7,
        ui.tags.script("""
        Shiny.addCustomMessageHandler('resize_plot', function(msg) {
            var el = document.getElementById('plot_curve');
            if (el) {
                el.style.height = msg.height + 'px';
                el.style.width  = msg.width  + 'px';
            }
        });
        """),
        ui.row(output_widget("plot_curve")),
        ui.div(
            {"class": "annotation-buttons-container"},
            ui.row(
                molt_columns,
                ui.column(
                    1,
                    ui.row(ui.input_action_button("set_arrest", "Arrest")),
                    ui.row(ui.input_action_button("set_death", "Dead")),
                    ui.row(ui.input_action_button("set_ignore_after", "Ignore After")),
                    ui.row(ui.input_action_button("set_ignore_point", "Ignore Point")),
                ),
                ui.row(
                    ui.column(
                        4,
                        ui.input_selectize(
                            "custom_column",
                            "Select custom column",
                            choices=custom_columns_choices,
                        ),
                    ),
                    ui.column(
                        4,
                        ui.input_text("new_custom_column", "Insert new custom column"),
                    ),
                    ui.column(
                        2,
                        ui.input_action_button("custom_annotation", "Annotate"),
                        ui.input_action_button("reset_custom_annotation", "Reset"),
                    ),
                    ui.column(
                        2,
                        ui.input_action_button(
                            "recompute_values_at_molt_point",
                            "Recompute values at molt (point)",
                        ),
                        ui.input_action_button(
                            "recompute_values_at_molt_all",
                            "Recompute values at molt (all points)",
                        ),
                    ),
                ),
                ui.row(
                    ui.input_slider(
                        "curve_plot_height",
                        "Plot height",
                        min=300,
                        max=1000,
                        step=10,
                        value=700,
                    ),
                    ui.input_slider(
                        "curve_plot_width",
                        "Plot width",
                        min=300,
                        max=2000,
                        step=10,
                        value=1300,
                    ),
                    ui.input_checkbox("log_scale", "Log scale", value=True),
                    ui.div(
                        ui.input_checkbox(
                            "recompute_values_at_molt_on_import",
                            "Recompute values at molt",
                            value=True,
                        ),
                        ui.input_file(
                            "import_file",
                            "Import Annotations",
                            accept=[".csv", ".mat", ".parquet"],
                            multiple=False,
                        ),
                    ),
                ),
                align="center",
            ),
        ),
        align="center",
    )


def create_timepoint_selector(
    list_channels,
    times,
    points,
    feature_columns,
    overlay_segmentation_choices,
    default_plotted_column,
):
    return ui.column(
        5,
        ui.output_ui("preload_progress"),
        ui.div({"class": "image-container"}, ui.output_ui("plot_image")),
        time_point_navigator("time", name="time", choices=times),
        time_point_navigator("point", name="point", choices=points),
        ui.div(
            {"class": "channel-selectors"},
            ui.input_selectize(
                "channel",
                "Select channel",
                choices=list_channels,
                selected=list_channels[1],
            ),
            ui.input_selectize(
                "channel_overlay",
                "Overlay channel",
                choices=list_channels,
                selected="None",
            ),
            ui.input_selectize(
                "segmentation_overlay",
                "Overlay segmentation",
                choices=overlay_segmentation_choices,
                selected="None",
            ),
        ),
        ui.row(
            ui.input_selectize(
                "column_to_plot",
                "Select column to plot",
                selected=default_plotted_column,
                choices=feature_columns,
            )
        ),
        ui.div(
            {"class": "save-row"},
            ui.input_action_button("save", "Save"),
            ui.input_dark_mode(id="color_mode", mode=None),
        ),
    )


def initialize_ui(filemap, recompute_features_at_molt=False):
    print("Initializing the UI ...")

    filemap = fix_experiment_time(filemap)

    times = (
        filemap.select(pl.col("Time"))
        .unique(maintain_order=True)
        .to_numpy()
        .squeeze()
        .tolist()
    )

    if not isinstance(times, list):
        times = [times]

    points = (
        filemap.select(pl.col("Point"))
        .unique(maintain_order=True)
        .to_numpy()
        .squeeze()
        .tolist()
    )

    if not isinstance(points, list):
        points = [points]

    (
        filemap,
        raw_column,
        feature_columns,
        custom_columns_choices,
        default_plotted_column,
        overlay_segmentation_choices,
    ) = populate_column_choices(filemap)

    filemap = process_feature_at_molt_columns(
        filemap,
        feature_columns,
        recompute_features_at_molt=recompute_features_at_molt,
    )

    n_channels = infer_n_channels(filemap, raw_column=raw_column)
    list_channels = ["None"] + [f"Channel {i+1}" for i in range(n_channels)]

    molt_annotator = create_molt_annotator(
        ECDYSIS_COLUMNS, MOLT_ENTRY_COLUMNS, custom_columns_choices
    )
    timepoint_selector = create_timepoint_selector(
        list_channels,
        times,
        points,
        feature_columns,
        overlay_segmentation_choices,
        default_plotted_column,
    )
    _css_path = Path(__file__).parent.parent / "static" / "custom.css"
    app_ui = ui.page_fluid(
        ui.include_css(_css_path, method="link_files"),
        ui.row(molt_annotator, timepoint_selector),
    )

    return (
        app_ui,
        filemap,
        raw_column,
        feature_columns,
        custom_columns_choices,
        points,
        times,
        default_plotted_column,
        n_channels,
    )
