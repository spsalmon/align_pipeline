import numpy as np
import polars as pl
from app_components.backend import MOLT_ENTRY_COLUMNS
from app_components.backend import _drop_join_artifact_columns
from app_components.backend import build_single_values_df
from app_components.backend import get_molt_interval_bands
from app_components.backend import open_filemap
from app_components.backend import populate_column_choices
from app_components.backend import process_feature_at_molt_columns
from app_components.backend import set_marker_shape


def make_minimal_filemap():
    """Filemap with only Time, Point, raw — as produced by a barely-processed experiment."""
    return pl.DataFrame(
        {
            "Time": [0, 1, 2, 0, 1, 2],
            "Point": [0, 0, 0, 1, 1, 1],
            "raw": [
                "/fake/0_0.tif",
                "/fake/0_1.tif",
                "/fake/0_2.tif",
                "/fake/1_0.tif",
                "/fake/1_1.tif",
                "/fake/1_2.tif",
            ],
        }
    )


# --- populate_column_choices ---


def test_populate_column_choices_creates_qc_placeholder():
    filemap = make_minimal_filemap()
    result_filemap, *_ = populate_column_choices(filemap)
    assert "placeholder_qc" in result_filemap.columns
    qc_values = result_filemap.select("placeholder_qc").to_numpy().squeeze()
    assert all(v == "worm" for v in qc_values)


def test_populate_column_choices_placeholder_feature_is_one():
    filemap = make_minimal_filemap()
    result_filemap, *_ = populate_column_choices(filemap)
    assert "placeholder_feature" in result_filemap.columns
    values = (
        result_filemap.select("placeholder_feature").to_numpy().squeeze().astype(float)
    )
    assert np.all(values == 1.0), f"Expected all 1.0, got {values}"


def test_populate_column_choices_feature_columns_not_empty():
    filemap = make_minimal_filemap()
    _, _, feature_columns, _, default_plotted_column, _ = populate_column_choices(
        filemap
    )
    assert len(feature_columns) > 0, "feature_columns must not be empty"
    assert "placeholder_feature" in feature_columns
    assert default_plotted_column == "placeholder_feature"


def test_populate_column_choices_segmentation_overlay_is_none_only():
    filemap = make_minimal_filemap()
    _, _, _, _, _, overlay_seg = populate_column_choices(filemap)
    assert overlay_seg == ["None"]


def test_populate_column_choices_does_not_treat_entry_columns_as_custom():
    """Molt-entry columns are canonical developmental-event columns and must not
    be misclassified as custom annotation columns. If they are, they leak into
    both work_df and the single-values df, producing a `_right` collision on
    save (polars DuplicateError)."""
    filemap = make_minimal_filemap()
    for col in MOLT_ENTRY_COLUMNS:
        filemap = filemap.with_columns(pl.lit(np.nan).alias(col))
    _, _, _, custom_columns_choices, _, _ = populate_column_choices(filemap)
    leaked = [c for c in MOLT_ENTRY_COLUMNS if c in custom_columns_choices]
    assert leaked == [], f"entry columns leaked into custom columns: {leaked}"


# --- _drop_join_artifact_columns ---


def test_drop_join_artifact_columns_removes_paired_right_column():
    filemap = make_minimal_filemap().with_columns(
        pl.lit(np.nan).alias("M1Entry"),
        pl.lit(np.nan).alias("M1Entry_right"),
    )
    healed = _drop_join_artifact_columns(filemap)
    assert "M1Entry_right" not in healed.columns
    assert "M1Entry" in healed.columns


def test_drop_join_artifact_columns_keeps_unpaired_right_column():
    """A `_right` column with no matching base is a legitimate name, not an artifact."""
    filemap = make_minimal_filemap().with_columns(pl.lit(1.0).alias("intensity_right"))
    healed = _drop_join_artifact_columns(filemap)
    assert "intensity_right" in healed.columns


# --- build_single_values_df ---


def test_build_single_values_df_without_ecdysis_columns():
    """Must not raise even though HatchTime, M1, M2, M3, M4 are absent."""
    filemap = make_minimal_filemap()
    result_filemap, *_ = populate_column_choices(filemap)
    df = build_single_values_df(result_filemap)
    assert "Point" in df.columns


def test_build_single_values_df_adds_ecdysis_columns_as_nan():
    filemap = make_minimal_filemap()
    result_filemap, *_ = populate_column_choices(filemap)
    df = build_single_values_df(result_filemap)
    for col in ["HatchTime", "M1", "M2", "M3", "M4"]:
        assert col in df.columns
        values = df.select(col).to_numpy().squeeze().astype(float)
        assert np.all(np.isnan(values)), f"{col} should be all NaN, got {values}"


def test_build_single_values_df_one_row_per_point():
    filemap = make_minimal_filemap()
    result_filemap, _, feature_columns, *_ = populate_column_choices(filemap)
    result_filemap = process_feature_at_molt_columns(result_filemap, feature_columns)
    df = build_single_values_df(result_filemap)
    n_points = filemap.select("Point").unique().height
    assert df.height == n_points


# --- process_feature_at_molt_columns ---


def test_process_feature_at_molt_columns_adds_ecdysis_columns():
    filemap = make_minimal_filemap()
    filemap, _, feature_columns, *_ = populate_column_choices(filemap)
    result = process_feature_at_molt_columns(filemap, feature_columns)
    for col in ["HatchTime", "M1", "M2", "M3", "M4"]:
        assert col in result.columns


def test_process_feature_at_molt_columns_adds_feature_at_molt_columns():
    filemap = make_minimal_filemap()
    filemap, _, feature_columns, *_ = populate_column_choices(filemap)
    result = process_feature_at_molt_columns(filemap, feature_columns)
    for ecdys in ["HatchTime", "M1", "M2", "M3", "M4"]:
        col = f"placeholder_feature_at_{ecdys}"
        assert col in result.columns


def test_process_feature_at_molt_columns_without_qc_column():
    """The annotation-import flow builds a fresh dataframe from the imported file
    and calls this function directly, without the placeholder_qc that
    populate_column_choices would otherwise inject. A filemap that carries a
    feature column but no qc column at all must not crash (regression: an empty
    qc_columns list produced a None column name)."""
    filemap = make_minimal_filemap().with_columns(
        pl.Series("ch2_seg_str_volume", [1.0, 2.0, 3.0, 1.5, 2.5, 3.5]),
        pl.Series("ExperimentTime", [0.0, 3600.0, 7200.0, 0.0, 3600.0, 7200.0]),
    )
    assert not [c for c in filemap.columns if "qc" in c]
    result = process_feature_at_molt_columns(filemap, ["ch2_seg_str_volume"])
    for ecdys in ["HatchTime", "M1", "M2", "M3", "M4"]:
        assert f"ch2_seg_str_volume_at_{ecdys}" in result.columns


def test_full_startup_pipeline_does_not_crash():
    """Simulate the full initialize_ui backend flow (minus infer_n_channels)."""
    filemap = make_minimal_filemap()
    (
        filemap,
        raw_column,
        feature_columns,
        custom_columns_choices,
        default_plotted_column,
        overlay_seg,
    ) = populate_column_choices(filemap)
    filemap = process_feature_at_molt_columns(filemap, feature_columns)
    single_values_df = build_single_values_df(filemap)

    assert raw_column == "raw"
    assert feature_columns == ["placeholder_feature"]
    assert default_plotted_column == "placeholder_feature"
    assert overlay_seg == ["None"]
    assert "Point" in single_values_df.columns


def test_build_single_values_df_adds_entry_columns_as_nan():
    filemap = make_minimal_filemap()
    result_filemap, *_ = populate_column_choices(filemap)
    df = build_single_values_df(result_filemap)
    for col in MOLT_ENTRY_COLUMNS:
        assert col in df.columns
        values = df.select(col).to_numpy().squeeze().astype(float)
        assert np.all(np.isnan(values)), f"{col} should be all NaN, got {values}"


def test_process_feature_at_molt_columns_adds_feature_at_entry_columns():
    filemap = make_minimal_filemap()
    filemap, _, feature_columns, *_ = populate_column_choices(filemap)
    result = process_feature_at_molt_columns(filemap, feature_columns)
    for entry in MOLT_ENTRY_COLUMNS:
        col = f"placeholder_feature_at_{entry}"
        assert col in result.columns


def open_legacy_event_filemap(tmp_path):
    """Open a filemap persisted by older pipelines, where never-populated event
    columns (e.g. the molt-entry columns) were stored as "" strings."""
    filemap, _, feature_columns, *_ = populate_column_choices(make_minimal_filemap())
    path = tmp_path / "analysis_filemap.parquet"
    filemap.with_columns(
        pl.lit("").alias("ExperimentTime"),
        pl.lit("").alias("M4Entry"),
        pl.lit("").alias(f"{feature_columns[0]}_at_M4Entry"),
    ).write_parquet(path)
    filemap, _ = open_filemap(str(path), open_annotated=False)
    return filemap, feature_columns


def test_process_feature_at_molt_columns_handles_legacy_event_columns(tmp_path):
    filemap, feature_columns = open_legacy_event_filemap(tmp_path)
    result = process_feature_at_molt_columns(filemap, feature_columns)
    column = f"{feature_columns[0]}_at_M4Entry"
    values = result.select(column).to_numpy().squeeze().astype(float)
    assert np.all(np.isnan(values))


def test_build_single_values_df_handles_legacy_event_columns(tmp_path):
    filemap, _ = open_legacy_event_filemap(tmp_path)
    df = build_single_values_df(filemap)
    assert df.schema["M4Entry"] == pl.Float64


def test_existing_ecdysis_columns_still_present():
    """Canonical developmental-event columns must remain untouched."""
    filemap = make_minimal_filemap()
    filemap, _, feature_columns, *_ = populate_column_choices(filemap)
    result = process_feature_at_molt_columns(filemap, feature_columns)
    for col in ["HatchTime", "M1", "M2", "M3", "M4"]:
        assert col in result.columns
        assert f"placeholder_feature_at_{col}" in result.columns


# --- get_molt_interval_bands ---


def test_get_molt_interval_bands_only_when_both_endpoints_finite():
    # M1: entry=1, exit=3 -> band; M2: entry=5, exit=nan -> no band;
    # M3: entry=nan, exit=9 -> no band; M4: entry=nan, exit=nan -> no band
    bands = get_molt_interval_bands(
        [1.0, 5.0, np.nan, np.nan],
        [3.0, np.nan, 9.0, np.nan],
        ["orange", "yellow", "green", "blue"],
    )
    assert len(bands) == 1
    band = bands[0]
    assert band["type"] == "rect"
    assert band["x0"] == 1.0 and band["x1"] == 3.0
    assert band["fillcolor"] == "orange"
    assert band["layer"] == "below"


def test_get_molt_interval_bands_orders_endpoints():
    bands = get_molt_interval_bands([7.0], [2.0], ["orange"])
    assert bands[0]["x0"] == 2.0 and bands[0]["x1"] == 7.0


def test_get_molt_interval_bands_tolerates_empty_string():
    bands = get_molt_interval_bands(["", 4.0], ["", 6.0], ["orange", "yellow"])
    assert len(bands) == 1
    assert bands[0]["fillcolor"] == "yellow"


# --- set_marker_shape ---


def test_set_marker_shape_marks_entry_as_diamond():
    times = np.array([0, 1, 2, 3, 4])
    qcs = np.array(["worm"] * 5)
    markers = set_marker_shape(
        times,
        selected_time_index=0,
        qcs=qcs,
        hatch_time=np.nan,
        m1=np.nan,
        m2=np.nan,
        m3=np.nan,
        m4=np.nan,
        m1_entry=2.0,
    )
    assert "diamond" in markers["symbol"]
