import os

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from align_toolbox.classification.qc_tools import compute_qc_features
from align_toolbox.foundation.file_handling import extract_time_point, write_filemap
from joblib import Parallel, delayed, load

from align_pipeline import utils


def main(input_pickle, output_file, block_config, config, filemap, n_jobs=-1):
    """Main function."""

    block_config = utils.load_pickles(block_config)[0]
    config = utils.load_pickles(config)[0]

    time_regex = config.get("time_regex", r"Time(\d+)")
    point_regex = config.get("point_regex", r"Point(\d+)")

    input_files = utils.load_pickles(input_pickle)[0]
    input_images = [f["image_path"] for f in input_files]
    input_masks = [f["mask_path"] for f in input_files]

    filemap = utils.load_pickles(filemap)[0]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    classifier_path = block_config["qc_model_path"]
    model_path_and_classes = load(classifier_path)
    mask_only = model_path_and_classes.get("mask_only", False)
    if mask_only:
        features = Parallel(n_jobs=n_jobs)(
            delayed(compute_qc_features)(mask_path, None) for mask_path in input_masks
        )
    else:
        features = Parallel(n_jobs=n_jobs)(
            delayed(compute_qc_features)(mask_path, image_path)
            for mask_path, image_path in zip(input_masks, input_images)
        )

    valid_indices = [i for i, x in enumerate(features) if x is not None]
    features = [features[i] for i in valid_indices]

    features_df = pd.concat(features, ignore_index=True)

    print(
        f"Ignored {(len(input_images) - len(valid_indices)) / len(input_images):.2%} of samples due to feature extraction failure / empty images."
    )

    _egg_model_relative = model_path_and_classes.get("egg_model_path", None)
    egg_model_path = (
        os.path.join(os.path.dirname(classifier_path), _egg_model_relative)
        if _egg_model_relative is not None
        else None
    )

    qc_model_path = os.path.join(
        os.path.dirname(classifier_path), model_path_and_classes["qc_model_path"]
    )
    import_eggs_from = block_config.get("qc_import_eggs_from", None)

    qc_classifier = xgb.XGBClassifier()
    qc_classifier.load_model(qc_model_path)
    qc_classes = model_path_and_classes["qc_classes"]

    # first, predict egg vs non-egg
    if egg_model_path is not None and import_eggs_from is None:
        egg_classifier = xgb.XGBClassifier()
        egg_classifier.load_model(egg_model_path)
        egg_classes = model_path_and_classes["egg_classes"]

        # if for some reason scikit-image decided to add features that we didn't include, we need to remove them
        egg_feature_names = egg_classifier.get_booster().feature_names
        extracted_feature_names = features_df.columns.to_list()
        extra_features = [
            fname for fname in extracted_feature_names if fname not in egg_feature_names
        ]

        egg_features_df = features_df.copy()

        if len(extra_features) > 0:
            egg_features_df = egg_features_df.drop(columns=extra_features)

        egg_features_dm = xgb.DMatrix(
            egg_features_df, feature_names=egg_features_df.columns.to_list()
        )
        egg_predictions = egg_classifier.get_booster().predict(egg_features_dm)
        # map numeric predictions back to class labels
        if len(egg_predictions.shape) > 1:
            egg_predictions = np.argmax(egg_predictions, axis=1)
        else:
            egg_predictions = (egg_predictions > 0.5).astype(int)
        egg_predictions = [egg_classes[pred] for pred in egg_predictions]

    elif import_eggs_from is not None:
        egg_predictions = filemap[import_eggs_from].to_list()
        egg_predictions = [egg_predictions[i] for i in valid_indices]

    else:
        egg_predictions = ["non-egg"] * len(features_df)

    non_egg_indices = [i for i, pred in enumerate(egg_predictions) if pred != "egg"]

    # QC-classify only the non-egg samples; with none (all eggs) skip the empty
    # DMatrix (which xgboost 3.4.1 chokes on) and keep every egg prediction.
    if non_egg_indices:
        qc_features_df = features_df.iloc[non_egg_indices].copy()

        # if for some reason scikit-image decided to add features that we didn't include, we need to remove them
        qc_feature_names = qc_classifier.get_booster().feature_names
        extracted_feature_names = qc_features_df.columns.to_list()
        extra_features = [
            fname for fname in extracted_feature_names if fname not in qc_feature_names
        ]
        if len(extra_features) > 0:
            qc_features_df = qc_features_df.drop(columns=extra_features)

        qc_features_dm = xgb.DMatrix(
            qc_features_df, feature_names=qc_features_df.columns.to_list()
        )
        qc_predictions = qc_classifier.get_booster().predict(qc_features_dm)
        # map numeric predictions back to class labels
        if len(qc_predictions.shape) > 1:
            qc_predictions = np.argmax(qc_predictions, axis=1)
        else:
            qc_predictions = (qc_predictions > 0.5).astype(int)
        qc_predictions = [qc_classes[pred] for pred in qc_predictions]
    else:
        qc_predictions = []

    # combine predictions - start with egg predictions
    predictions = egg_predictions.copy()
    # overwrite non-egg positions with qc predictions
    for i, idx in enumerate(non_egg_indices):
        predictions[idx] = qc_predictions[i]

    # map predictions back to original image indices
    full_predictions = np.full_like(
        np.array(input_masks), fill_value="error", dtype=object
    )
    for idx, valid_idx in enumerate(valid_indices):
        full_predictions[valid_idx] = predictions[idx]

    # time_points = Parallel(n_jobs=-1)(
    #     delayed(extract_time_point)(path, time_regex, point_regex) for path in input_masks
    # )

    col_name = os.path.splitext(os.path.basename(output_file))[0]

    rows = []
    for i, path in enumerate(input_masks):
        try:
            time_point = extract_time_point(path, time_regex, point_regex)
            rows.append(
                {
                    "Time": time_point[0],
                    "Point": time_point[1],
                    col_name: full_predictions[i],
                }
            )
        except ValueError:
            print(f"Could not extract time and point from path: {path}. Skipping.")

    worm_types_dataframe = pl.DataFrame(rows)

    write_filemap(worm_types_dataframe, output_file)


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(
        args.input,
        args.output,
        args.block_config,
        args.config,
        args.filemap,
        args.n_jobs,
    )
