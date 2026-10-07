import numpy as np
import polars as pl
from app_components.backend import (
    ECDYSIS_COLUMNS,
    build_single_values_df,
    clear_values_at_missing_events,
    merge_imported_annotations,
    process_feature_at_molt_columns,
    recompute_values_at_molt_of_point,
)

N_POINTS, N_TIMES = 3, 40
ANNOTATION_COLUMNS = ECDYSIS_COLUMNS + ["Death", "Ignore", "Arrest"]


def make_gui_filemap():
    rng = np.random.default_rng(0)
    rows = [
        {
            "Point": p,
            "Time": t,
            "ExperimentTime": float(t * 600),
            "volume": float(1000 * np.exp(0.05 * t) + rng.normal(0, 20)),
            "qc": "egg" if t < 3 else "worm",
        }
        for p in range(N_POINTS)
        for t in range(N_TIMES)
    ]
    filemap = pl.DataFrame(rows)
    return filemap.with_columns(
        *[pl.lit(None, dtype=pl.Float64).alias(e) for e in ECDYSIS_COLUMNS],
        *[
            pl.lit(None, dtype=pl.Float64).alias(f"volume_at_{e}")
            for e in ECDYSIS_COLUMNS
        ],
    )


def make_imported_filemap(gui_filemap):
    """Annotated file with molts set but stale values at molt and features
    from a different segmentation."""
    return gui_filemap.with_columns(
        pl.lit(5.0).alias("HatchTime"),
        pl.lit(20.0).alias("M1"),
        pl.lit(123.0).alias("volume_at_HatchTime"),
        pl.lit(456.0).alias("volume_at_M1"),
        (pl.col("volume") * 3).alias("volume"),
        pl.lit("error").alias("qc"),
    )


def recompute_with_button(gui_filemap, single_values_df):
    point_filemaps = gui_filemap.partition_by("Point", maintain_order=True)
    return pl.concat(
        [
            recompute_values_at_molt_of_point(point_filemap, point_values)
            for point_filemap, point_values in zip(
                point_filemaps,
                single_values_df.partition_by("Point", maintain_order=True),
            )
        ]
    )


def merge_and_build(gui_filemap, imported_df):
    merged = merge_imported_annotations(
        gui_filemap, imported_df, ["volume"], ANNOTATION_COLUMNS
    )
    return build_single_values_df(merged).sort("Point")


def test_import_matches_recompute_button():
    gui_filemap = make_gui_filemap()
    imported = merge_and_build(gui_filemap, make_imported_filemap(gui_filemap))

    recomputed = recompute_with_button(gui_filemap, imported)

    for column in ["volume_at_HatchTime", "volume_at_M1"]:
        np.testing.assert_allclose(
            imported[column].to_numpy(), recomputed[column].to_numpy()
        )
    assert not np.isclose(imported["volume_at_HatchTime"].to_numpy(), 123.0).any()


def test_import_matches_rows_on_point_and_time_not_position():
    gui_filemap = make_gui_filemap()
    imported_df = make_imported_filemap(gui_filemap)

    ordered = merge_and_build(gui_filemap, imported_df)
    shuffled = merge_and_build(gui_filemap, imported_df.sample(fraction=1.0, seed=1))

    np.testing.assert_allclose(
        ordered["volume_at_M1"].to_numpy(), shuffled["volume_at_M1"].to_numpy()
    )


def test_import_only_returns_imported_points_and_clears_missing_events():
    gui_filemap = make_gui_filemap().with_columns(
        pl.lit(30.0).alias("M2"), pl.lit(999.0).alias("volume_at_M2")
    )
    imported_df = make_imported_filemap(make_gui_filemap()).filter(pl.col("Point") != 2)

    imported = merge_and_build(gui_filemap, imported_df)

    assert imported["Point"].to_list() == [0, 1]
    assert imported["M2"].is_nan().all()
    assert imported["volume_at_M2"].is_nan().all()


def test_clear_values_at_missing_events_only_clears_events_without_time():
    filemap = pl.DataFrame(
        {
            "Point": [0, 1],
            "M1": [10, None],
            "M1Entry": [None, 8.0],
            "volume_at_M1": [1.0, 2.0],
            "volume_at_M1Entry": [3.0, 4.0],
        }
    )

    cleared = clear_values_at_missing_events(filemap)

    np.testing.assert_array_equal(cleared["volume_at_M1"].to_numpy(), [1.0, np.nan])
    np.testing.assert_array_equal(
        cleared["volume_at_M1Entry"].to_numpy(), [np.nan, 4.0]
    )


def test_import_without_recompute_clears_stale_values_of_missing_events():
    gui_filemap = make_gui_filemap()
    imported_df = make_imported_filemap(gui_filemap).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("M1")
    )

    processed = process_feature_at_molt_columns(
        imported_df, ["volume"], recompute_features_at_molt=False
    )
    single_values = build_single_values_df(clear_values_at_missing_events(processed))

    assert single_values["volume_at_M1"].is_nan().all()
    np.testing.assert_allclose(single_values["volume_at_HatchTime"].to_numpy(), 123.0)
