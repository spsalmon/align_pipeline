import os
import warnings

import numpy as np
import pandas as pd
import polars as pl
import torch
from joblib import Parallel
from joblib import delayed
from joblib import parallel_config
from torch.utils.data import DataLoader
from towbintools.data_analysis import compute_instantaneous_growth_rate_classified
from towbintools.data_analysis import compute_series_at_time_classified
from towbintools.data_analysis import correct_series_with_classification
from towbintools.deep_learning.deep_learning_tools import (
    load_keypoint_detection_model_from_checkpoint,
)
from towbintools.deep_learning.utils.dataset import KeypointDetection1DPredictionDataset
from towbintools.foundation import detect_molts
from towbintools.foundation.detect_molts import find_hatch_time
from towbintools.foundation.file_handling import read_filemap
from towbintools.foundation.file_handling import write_filemap
from towbintools.foundation.keypoint_detection import heatmap_to_keypoints_1D
from towbintools.foundation.utils import find_best_string_match
from towbintools.foundation.worm_features import get_features_to_compute_at_molt
from tqdm import tqdm

from towbintools_pipeline import utils

FEATURES_TO_COMPUTE_AT_MOLT = get_features_to_compute_at_molt()


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


def run_detect_molts(
    analysis_filemap,
    volume_column,
    point,
    molt_size_range=[6.6e4, 15e4, 36e4, 102e4],
    search_width=20,
    fit_width=5,
):
    data_of_point = analysis_filemap.filter(pl.col("Point") == point)
    volumes = data_of_point.select(pl.col(volume_column))

    volumes = volumes.replace("", np.nan)
    volumes = volumes.to_numpy().squeeze().astype(float)

    # find the qc column with the best match to the volume column
    qc_columns = [col for col in analysis_filemap.columns if "qc" in col]

    if len(qc_columns) == 1:
        worm_type_column = qc_columns[0]
    else:
        worm_type_column = find_best_string_match(volume_column, qc_columns)

    worm_types = data_of_point.select(pl.col(worm_type_column)).to_numpy().squeeze()
    try:
        # Detect molts
        ecdysis = detect_molts.find_molts(
            volumes, worm_types, molt_size_range, search_width, fit_width
        )
    except ValueError as e:
        print(f"Error in point {point}: {e}")
        # No molt detected
        ecdysis = {
            "hatch_time": np.nan,
            "M1": np.nan,
            "M2": np.nan,
            "M3": np.nan,
            "M4": np.nan,
        }

    print(f"Point {point} done, ecdysis: {ecdysis}")

    return {
        "Point": point,
        "HatchTime": ecdysis["hatch_time"],
        "M1": ecdysis["M1"],
        "M2": ecdysis["M2"],
        "M3": ecdysis["M3"],
        "M4": ecdysis["M4"],
    }


def run_detect_molts_deep_learning(
    analysis_filemap,
    molt_detection_columns,
    model_path,
    batch_size=1,
):
    # find the qc column with the best match to the volume column
    qc_columns = [col for col in analysis_filemap.columns if "qc" in col]

    if len(qc_columns) == 1:
        worm_type_column = qc_columns[0]
    else:
        worm_type_column = find_best_string_match(molt_detection_columns[0], qc_columns)

    worm_type_data = separate_column_by_point(analysis_filemap, worm_type_column)

    hatch_indices = []
    for i, worm_type_series in enumerate(worm_type_data):
        hatch_time = find_hatch_time(
            worm_type_series,
        )
        hatch_indices.append(hatch_time)
    hatch_indices = np.array(hatch_indices)

    features = []
    for column in molt_detection_columns:
        if column not in analysis_filemap.columns:
            raise ValueError(f"Column {column} not found in analysis filemap.")
        column_data = separate_column_by_point(analysis_filemap, column)

        for i, series_i in enumerate(column_data):
            worm_type_i = worm_type_data[i]
            series_length = series_i.shape[-1]

            log_volume_data = np.log(
                correct_series_with_classification(series_i, worm_type_i)
            )

            log_volume_growth_rate = compute_instantaneous_growth_rate_classified(
                log_volume_data,
                time=np.linspace(0, series_length - 1, series_length),
                qc=worm_type_i,
                lmbda=0.005,
                order=2,
                medfilt_window=5,
            )

            features.append(np.stack([log_volume_data, log_volume_growth_rate], axis=0))

    molt_detection_data = np.stack(features, axis=0)
    if molt_detection_data.ndim == 1:
        molt_detection_data = molt_detection_data[np.newaxis, ...]

    print(f"Molt detection data shape: {molt_detection_data.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_keypoint_detection_model_from_checkpoint(model_path).to(device)
    model.eval()

    dataset = KeypointDetection1DPredictionDataset(
        inputs=molt_detection_data,
        resize_method="pad",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    molts_indices = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, valid_mask, invalid_series_index, original_shapes = batch
            x = x.to(device)
            valid_mask = valid_mask.to(device)
            heatmaps, presences = model(x, mask=valid_mask)
            heatmaps = heatmaps.cpu().numpy()
            presences = presences.cpu().numpy()

            for j, (heatmap, presence) in enumerate(zip(heatmaps, presences)):
                if j in invalid_series_index:
                    molts_indices.append(
                        [np.nan] * 4
                    )  # series was invalid so no molts could be detected
                else:
                    m = heatmap_to_keypoints_1D(heatmap, presence)
                    print(f"Molts detected : {m}")
                    molts_indices.append(m)

    molts_indices = np.array(molts_indices)

    # Create a DataFrame with the results
    points = (
        analysis_filemap.select(pl.col("Point").unique(maintain_order=True))
        .to_numpy()
        .flatten()
    )
    times = separate_column_by_point(analysis_filemap, "Time")

    hatch_times = []
    molt_times = []
    for i, time in enumerate(times):
        hatch = hatch_indices[i]
        molts = molts_indices[i]
        molt_times_of_point = []
        if np.isnan(hatch):
            hatch_times.append(np.nan)
        else:
            hatch_times.append(time[int(hatch)])
        for m in molts:
            if np.isnan(m):
                molt_times_of_point.append(np.nan)
            else:
                molt_times_of_point.append(time[int(m)])
        molt_times.append(molt_times_of_point)

    molt_times = np.array(molt_times)

    molt_columns = ["M1", "M2", "M3", "M4"]
    molt_data = {
        "Point": points,
        "HatchTime": hatch_times,
    }
    for i, column in enumerate(molt_columns):
        molt_data[column] = molt_times[:, i]
    molts_dataframe = pl.DataFrame(molt_data)
    return molts_dataframe


def compute_features_at_molt(analysis_filemap, molt_dataframe, point):
    data_of_point = analysis_filemap.filter(pl.col("Point") == point)
    molt_data_of_point = molt_dataframe.filter(pl.col("Point") == point)
    data_of_point = data_of_point.sort(by="Time")

    columns_to_compute = []
    for feature in FEATURES_TO_COMPUTE_AT_MOLT:
        columns_to_compute.extend(
            [
                column
                for column in data_of_point.columns
                if (feature in column) and ("at_" not in column)
            ]
        )

    molt_values = (
        molt_data_of_point.select(pl.col(["HatchTime", "M1", "M2", "M3", "M4"]))
        .to_numpy()
        .squeeze()
    )
    time_values = data_of_point.select(pl.col("Time")).to_numpy().squeeze()
    # compute the features at each molt
    features_at_molt = {"Point": point}
    qc_columns = [col for col in analysis_filemap.columns if "qc" in col]

    for column in columns_to_compute:
        # find the qc column with the best match to the volume column
        if len(qc_columns) == 1:
            worm_type_column = qc_columns[0]
        else:
            worm_type_column = find_best_string_match(column, qc_columns)

        print(f"Using worm type column {worm_type_column} for feature column {column}")

        worm_types = data_of_point.select(pl.col(worm_type_column)).to_numpy().squeeze()
        column_data = data_of_point.select(pl.col(column))
        column_data = column_data.to_numpy().squeeze().astype(float)

        for i, molt in enumerate(["HatchTime", "M1", "M2", "M3", "M4"]):
            molt_time = float(molt_values[i])
            try:
                if not np.isnan(molt_time):
                    features_at_molt[f"{column}_at_{molt}"] = (
                        compute_series_at_time_classified(
                            column_data,
                            molt_time,
                            time_values,
                            worm_types,
                        ).item()
                    )
                else:
                    features_at_molt[f"{column}_at_{molt}"] = np.nan
            except Exception as e:
                print(f"Error in point {point}, column {column}, molt {molt}: {e}")
                features_at_molt[f"{column}_at_{molt}"] = np.nan

    return features_at_molt


def main(input_dataframe_path, output_file, block_config, config, n_jobs):
    """Main function."""

    block_config = utils.load_pickles(block_config)[0]
    config = utils.load_pickles(config)[0]

    method = block_config.get("molt_detection_method", "legacy")

    if method == "deep_learning":
        model_path = block_config.get("molt_detection_model_path", None)
        if model_path is None:
            raise ValueError("Model path must be provided for deep learning method.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

    if (
        "molt_detection_columns" not in block_config
        and "molt_detection_volume" in block_config
    ):
        warnings.warn(
            "The option 'molt_detection_volume' is deprecated and will be removed in a future version. Use 'molt_detection_columns' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if "molt_detection_columns" not in block_config:
        molt_detection_columns = [block_config["molt_detection_volume"]]
    else:
        molt_detection_columns = block_config["molt_detection_columns"]

    analysis_filemap = utils.load_pickles(input_dataframe_path)[0]
    analysis_filemap = read_filemap(analysis_filemap)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if method == "legacy":
        with parallel_config(backend="loky", n_jobs=n_jobs):
            molts = Parallel()(
                delayed(run_detect_molts)(
                    analysis_filemap,
                    molt_detection_columns[0],
                    point,
                )
                for point in analysis_filemap["Point"].unique(maintain_order=True)
            )
            molts_dataframe = pd.DataFrame(molts)
    elif method == "deep_learning":
        molts_dataframe = run_detect_molts_deep_learning(
            analysis_filemap,
            molt_detection_columns,
            model_path,
            batch_size=block_config.get("molt_detection_batch_size", 1),
        )

    # compute other features at each molt
    with parallel_config(backend="loky", n_jobs=n_jobs):
        other_features_at_molt = Parallel()(
            delayed(compute_features_at_molt)(
                analysis_filemap,
                molts_dataframe,
                point,
            )
            for point in analysis_filemap["Point"].unique(maintain_order=True)
        )

    other_features_at_molt_dataframe = pl.DataFrame(other_features_at_molt)
    molts_dataframe = molts_dataframe.join(
        other_features_at_molt_dataframe,
        on="Point",
        how="left",
    )

    write_filemap(molts_dataframe, output_file)


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.block_config, args.config, args.n_jobs)
