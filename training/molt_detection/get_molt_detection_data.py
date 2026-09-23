import argparse
import os
import pickle

import numpy as np
import polars as pl
import yaml
from align_toolbox.data_analysis.growth_rate import (
    compute_instantaneous_growth_rate_classified,
)
from align_toolbox.data_analysis.time_series import correct_series_with_classification
from align_toolbox.foundation.file_handling import read_filemap
from tqdm import tqdm

# Internal feature-engineering parameters for the instantaneous growth rate.
# These are implementation details of how the log-volume series is smoothed and
# differentiated, not user-facing knobs, so they are kept as module constants.
GROWTH_RATE_LAMBDA = 0.005
GROWTH_RATE_ORDER = 2
GROWTH_RATE_MEDFILT_WINDOW = 5


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    args = parser.parse_args()
    return args


def landmarks_to_gaussian_keymap(landmarks, length, sigma_frames=3.0):
    landmarks = np.asarray(landmarks, dtype=float)
    nan_mask = np.isnan(landmarks)
    centers = np.where(nan_mask, 0.0, landmarks)[:, None]
    x = np.arange(length, dtype=float)[None, :]

    keymap = np.exp(-0.5 * ((x - centers) / sigma_frames) ** 2)
    keymap[nan_mask] = 0.0
    return keymap


def get_analysis_filemap(experiment_path):
    analysis_dirs = [
        os.path.join(experiment_path, d)
        for d in os.listdir(experiment_path)
        if os.path.isdir(os.path.join(experiment_path, d))
        and "analysis" in d
        and "matlab" not in d
    ]
    report_dirs = [
        report
        for d in analysis_dirs
        if os.path.isdir(report := os.path.join(d, "report"))
    ]

    for report_dir in report_dirs:
        filemaps = [
            os.path.join(report_dir, f)
            for f in os.listdir(report_dir)
            if "analysis_filemap_annotated" in f
        ]
        if filemaps:
            return max(filemaps, key=os.path.getmtime)
    return None


def get_features_and_ground_truth_molts(
    filemap,
    worm_type_column="ch2_seg_str_worm_type",
    volume_column="ch2_seg_str_volume",
    sigma_frames=3.0,
    min_series_length=200,
    max_series_length=1000,
):
    X = []
    y = []
    keypoints_all = []

    if worm_type_column not in filemap.columns:
        worm_type_column = "ch2_seg_str_qc"

    for point_data in filemap.partition_by("Point", maintain_order=True):
        point_data = point_data.sort("Time")
        point = point_data["Point"][0]

        worm_type_data = point_data[worm_type_column].to_numpy()
        if np.all(worm_type_data == worm_type_data[0]):
            print(f"Skipping point {point}: all worm types are {worm_type_data[0]}")
            continue

        keypoints = point_data.select(
            pl.col(["HatchTime", "M1", "M2", "M3", "M4"]).cast(pl.Float64, strict=False)
        ).to_numpy()[0]
        if np.all(np.isnan(keypoints)):
            print(f"Skipping point {point}: all keypoints are NaN")
            continue
        keypoints = keypoints[1:]

        volume_data = point_data[volume_column].to_numpy()
        series_length = volume_data.shape[-1]
        if not min_series_length <= series_length <= max_series_length:
            print(
                f"Skipping point {point}: length {series_length} outside "
                f"[{min_series_length}, {max_series_length}]"
            )
            continue

        log_volume_data = np.log(
            correct_series_with_classification(volume_data, worm_type_data)
        )

        log_volume_growth_rate = compute_instantaneous_growth_rate_classified(
            log_volume_data,
            time=np.linspace(0, series_length - 1, series_length),
            qc=worm_type_data,
            lmbda=GROWTH_RATE_LAMBDA,
            order=GROWTH_RATE_ORDER,
            medfilt_window=GROWTH_RATE_MEDFILT_WINDOW,
        )

        features = np.stack([log_volume_data, log_volume_growth_rate], axis=0)

        X.append(features)
        y.append(
            landmarks_to_gaussian_keymap(
                keypoints, length=series_length, sigma_frames=sigma_frames
            )
        )
        keypoints_all.append(keypoints)

    return X, y, keypoints_all


def get_all_features_and_ground_truth_molts(
    filemaps,
    worm_type_column="ch2_seg_str_worm_type",
    volume_column="ch2_seg_str_volume",
    sigma_frames=3.0,
    min_series_length=200,
    max_series_length=1000,
):
    X = []
    y = []
    keypoints_all = []

    for filemap_path in tqdm(filemaps):
        print(f"Processing filemap: {filemap_path}")
        try:
            X_i, y_i, keypoints_i = get_features_and_ground_truth_molts(
                read_filemap(filemap_path),
                worm_type_column=worm_type_column,
                volume_column=volume_column,
                sigma_frames=sigma_frames,
                min_series_length=min_series_length,
                max_series_length=max_series_length,
            )
            X.extend(X_i)
            y.extend(y_i)
            keypoints_all.extend(keypoints_i)
        except Exception as e:
            print(f"Error processing filemap {filemap_path}: {e}")

    return X, y, np.array(keypoints_all)


def find_valid_filemaps(
    storage_cluster_path="/mnt/towbin.data/shared",
    experimentalists=("kstojanovski", "plenart", "igheor", "spsalmon"),
    scopes=("ti2", "orca", "lipsi", "crest", "squid"),
    volume_column="ch2_seg_str_volume",
    min_year=2024,
    magnifications=("10x", "10X"),
):
    scope_variations = {
        variation
        for scope in scopes
        for variation in (scope, scope.upper(), scope.capitalize())
    }

    filemaps = []
    for experimentalist in experimentalists:
        exp_dir = os.path.join(storage_cluster_path, experimentalist)
        for name in os.listdir(exp_dir):
            experiment = os.path.join(exp_dir, name)
            if not os.path.isdir(experiment):
                continue

            try:
                if int(name[:4]) < min_year:
                    continue
            except ValueError:
                continue

            if not any(magnification in name for magnification in magnifications):
                continue
            if not any(scope in name for scope in scope_variations):
                continue

            filemap_path = get_analysis_filemap(experiment)
            if filemap_path is None:
                continue

            f = read_filemap(filemap_path)
            if volume_column not in f.columns:
                continue
            filemaps.append(filemap_path)

    return filemaps


if __name__ == "__main__":
    config_file = get_args().config
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    storage_cluster_path = config.get("storage_cluster_path", "/mnt/towbin.data/shared")
    experimentalists = config.get(
        "experimentalists", ["kstojanovski", "plenart", "igheor", "spsalmon"]
    )
    scopes = config.get("scopes", ["ti2", "orca", "lipsi", "crest", "squid"])
    min_year = config.get("min_year", 2024)
    magnifications = config.get("magnifications", ["10x", "10X"])
    worm_type_column = config.get("worm_type_column", "ch2_seg_str_worm_type")
    volume_column = config.get("volume_column", "ch2_seg_str_volume")
    sigma_frames = config.get("sigma_frames", 2.0)
    keymap_type = config.get("keymap_type", "gaussian")
    min_series_length = config.get("min_series_length", 200)
    max_series_length = config.get("max_series_length", 1000)
    output_dir = config.get("output_dir", "./molt_detection_dataset")

    os.makedirs(output_dir, exist_ok=True)

    filemaps = find_valid_filemaps(
        storage_cluster_path=storage_cluster_path,
        experimentalists=experimentalists,
        scopes=scopes,
        volume_column=volume_column,
        min_year=min_year,
        magnifications=magnifications,
    )
    print(f"Found {len(filemaps)} valid experiments")

    X, y, keypoints_all = get_all_features_and_ground_truth_molts(
        filemaps,
        worm_type_column=worm_type_column,
        volume_column=volume_column,
        sigma_frames=sigma_frames,
        min_series_length=min_series_length,
        max_series_length=max_series_length,
    )

    pickle.dump(X, open(os.path.join(output_dir, "X_molt_detection.pickle"), "wb"))
    pickle.dump(
        y,
        open(os.path.join(output_dir, f"y_{keymap_type}_molt_detection.pickle"), "wb"),
    )
    pickle.dump(
        keypoints_all,
        open(os.path.join(output_dir, "keypoints_molt_detection.pickle"), "wb"),
    )
