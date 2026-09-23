import polars as pl
import pytest
from align_toolbox.foundation.file_handling import read_filemap, write_filemap


def make_legacy_filemap():
    """Filemap as persisted by older pipelines: missing values filled with ""."""
    return pl.DataFrame(
        {
            "Time": [0, 1, 2],
            "Point": [0, 0, 0],
            "raw": ["/fake/0.tif", "/fake/1.tif", "/fake/2.tif"],
            "analysis/ch2_seg": ["/fake/0_seg.tif", "", "/fake/2_seg.tif"],
            "ch2_seg_str_width_skew": ["0.5", "", "1.5"],
            "ExperimentTime": ["0", "600", ""],
            "ch2_seg_str_qc": ["worm", "egg", ""],
        }
    )


@pytest.mark.parametrize("extension", ["parquet", "csv"])
def test_read_filemap_heals_legacy_empty_strings(tmp_path, extension):
    path = str(tmp_path / f"analysis_filemap.{extension}")
    getattr(make_legacy_filemap(), f"write_{extension}")(path)

    filemap = read_filemap(path)

    assert filemap.schema["ch2_seg_str_width_skew"] == pl.Float64
    assert filemap.schema["ExperimentTime"] == pl.Int64
    assert filemap.schema["ch2_seg_str_qc"] == pl.String
    assert filemap.schema["analysis/ch2_seg"] == pl.String
    assert filemap["ch2_seg_str_width_skew"].to_list() == [0.5, None, 1.5]
    assert filemap["analysis/ch2_seg"].to_list()[1] is None
    assert filemap["ch2_seg_str_qc"].to_list() == ["worm", "egg", None]


def test_read_filemap_converts_nan_to_null(tmp_path):
    path = str(tmp_path / "analysis_filemap.parquet")
    pl.DataFrame({"Point": [0, 1], "volume": [1.0, float("nan")]}).write_parquet(path)

    assert read_filemap(path)["volume"].to_list() == [1.0, None]


def test_csv_and_parquet_read_back_identically(tmp_path):
    filemap = make_legacy_filemap().with_columns(
        pl.Series("volume", [1.0, float("nan"), None])
    )
    write_filemap(filemap, str(tmp_path / "f.parquet"))
    write_filemap(filemap, str(tmp_path / "f.csv"))

    assert read_filemap(str(tmp_path / "f.parquet")).equals(
        read_filemap(str(tmp_path / "f.csv"))
    )


def test_write_filemap_never_persists_empty_strings_or_nan(tmp_path):
    path = str(tmp_path / "analysis_filemap.parquet")
    filemap = make_legacy_filemap().with_columns(
        pl.Series("volume", [1.0, float("nan"), 2.0])
    )
    write_filemap(filemap, path)

    persisted = pl.read_parquet(path)
    assert persisted.schema["ch2_seg_str_width_skew"] == pl.Float64
    assert persisted["volume"].null_count() == 1
    assert persisted.filter(pl.col("ch2_seg_str_qc") == "").is_empty()
