import argparse
import os
import shutil

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from align_toolbox.classification.qc_tools import compute_qc_features
from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed, dump
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    args = parser.parse_args()
    return args


def get_scope_pixelsize(image_path, pixelsizes):
    """Return the pixel size of the scope an image was acquired on.

    The scope is read from the image name, which ``get_qc_data.py`` saves as
    ``image_{i}_{class}_{scope}.tiff``. Returns ``None`` when ``pixelsizes`` is
    not set, so features stay in pixels as for models trained before pixel sizes
    were taken into account.
    """
    if pixelsizes is None:
        return None
    stem = os.path.splitext(os.path.basename(image_path))[0]
    # longest match first, so that e.g. "ti2_fast" wins over "fast"
    for scope in sorted(pixelsizes, key=len, reverse=True):
        if stem.endswith(f"_{scope}"):
            return float(pixelsizes[scope])
    raise ValueError(
        f"Could not find the scope of {image_path} in pixelsizes "
        f"({list(pixelsizes)}). Add its scope to the pixelsizes of the config."
    )


def cached_features_match(processed_annotation_csv_path, pixelsizes):
    """Check that cached features were computed with the current pixel sizes."""
    cached_df = pd.read_csv(processed_annotation_csv_path)
    if "PixelSize" not in cached_df.columns:
        return pixelsizes is None
    expected = cached_df["ImagePath"].apply(
        lambda x: get_scope_pixelsize(x, pixelsizes)
    )
    return np.allclose(
        expected.astype(float), cached_df["PixelSize"].astype(float), equal_nan=True
    )


def load_dataset(dataset_path, mask_only, pixelsizes, rerun_feature_extraction):
    """Load the annotations of one dataset and compute (or reuse) its features.

    Features are cached inside the dataset directory, so each dataset of a
    combined training keeps its own cache.
    """
    project_yaml = os.path.join(dataset_path, "project.yaml")
    with open(project_yaml) as f:
        project_config = yaml.safe_load(f)

    classes = project_config["classes"]
    mask_dir = os.path.join(dataset_path, "masks")
    annotation_csv_path = os.path.join(dataset_path, "annotations", "annotations.csv")
    annotations_df = pd.read_csv(annotation_csv_path)
    # normalise Windows-style paths coming from the annotation tool
    annotations_df["ImagePath"] = annotations_df["ImagePath"].str.replace("\\", "/")
    annotations_df["ImagePath"] = annotations_df["ImagePath"].str.replace(
        "//izbkingston", "/mnt"
    )
    # add the mask paths to the dataframe
    annotations_df["MaskPath"] = annotations_df["ImagePath"].apply(
        lambda x: os.path.join(mask_dir, os.path.basename(x))
    )
    # remove rows with no class label or classes not in the project config
    annotations_df = annotations_df[annotations_df["Class"].isin(classes)].reset_index(
        drop=True
    )

    print(annotations_df.head())
    # check that image and mask paths exist
    for idx, row in annotations_df.iterrows():
        assert os.path.exists(
            row["ImagePath"]
        ), f"Image path does not exist: {row['ImagePath']}"
        assert os.path.exists(
            row["MaskPath"]
        ), f"Mask path does not exist: {row['MaskPath']}"

    features_df_path = os.path.join(dataset_path, "features.csv")
    processed_annotation_csv_path = os.path.join(
        dataset_path, "processed_annotations.csv"
    )
    use_cached_features = (
        os.path.exists(features_df_path)
        and os.path.exists(processed_annotation_csv_path)
        and not rerun_feature_extraction
    )
    if use_cached_features and not cached_features_match(
        processed_annotation_csv_path, pixelsizes
    ):
        print("Cached features were computed with other pixel sizes, recomputing.")
        use_cached_features = False

    if use_cached_features:
        annotations_df = pd.read_csv(processed_annotation_csv_path)
        features_df = pd.read_csv(features_df_path)
    else:
        image_pixelsizes = [
            get_scope_pixelsize(image_path, pixelsizes)
            for image_path in annotations_df["ImagePath"]
        ]
        annotations_df["PixelSize"] = image_pixelsizes
        X = Parallel(n_jobs=-1)(
            delayed(compute_qc_features)(
                row["MaskPath"],
                None if mask_only else row["ImagePath"],
                pixelsize=pixelsize,
            )
            for (_, row), pixelsize in tqdm(
                zip(annotations_df.iterrows(), image_pixelsizes),
                total=annotations_df.shape[0],
            )
        )
        valid_indices = [i for i, x in enumerate(X) if x is not None]
        X = [X[i] for i in valid_indices]
        annotations_df = annotations_df.iloc[valid_indices].reset_index(drop=True)
        features_df = pd.concat(X, ignore_index=True)
        features_df.to_csv(features_df_path, index=False)
        annotations_df.to_csv(processed_annotation_csv_path, index=False)

    annotations_df["Dataset"] = dataset_path
    return annotations_df, features_df


def train_xgb_model(
    train_X,
    train_y,
    test_X,
    test_y,
    classes,
    optimize_hyperparameters,
    n_points=10,
    n_iter=50,
):
    w_train = compute_sample_weight(class_weight="balanced", y=train_y)
    if len(classes) == 2:
        objective = "binary:logistic"
        eval_metric = "logloss"
        num_class = None
    else:
        objective = "multi:softprob"
        eval_metric = "mlogloss"
        num_class = len(classes)

    if optimize_hyperparameters:

        def xgb_eval(
            max_depth, learning_rate, colsample_bytree, n_estimators, min_child_weight
        ):
            params = {
                "objective": objective,
                "num_class": num_class,
                "eval_metric": eval_metric,
                "max_depth": int(max_depth),
                "learning_rate": learning_rate,
                "colsample_bytree": colsample_bytree,
                "n_estimators": int(n_estimators),
                "min_child_weight": min_child_weight,
                "random_state": 42,
            }

            model = xgb.XGBClassifier(**params)
            model.fit(train_X, train_y, verbose=False, sample_weight=w_train)
            score = model.score(test_X, test_y)
            return score

        pbounds = {
            "max_depth": (3, 10),
            "learning_rate": (0.01, 0.3),
            "colsample_bytree": (0.5, 1.0),
            "n_estimators": (50, 300),
            "min_child_weight": (1, 10),
        }

        optimizer = BayesianOptimization(f=xgb_eval, pbounds=pbounds, random_state=42)
        optimizer.maximize(init_points=n_points, n_iter=n_iter)

        # use the best hyperparameters to train the final model
        best_params = optimizer.max["params"]
        params = {
            "objective": objective,
            "num_class": num_class,
            "eval_metric": eval_metric,
            "max_depth": int(best_params["max_depth"]),
            "learning_rate": best_params["learning_rate"],
            "colsample_bytree": best_params["colsample_bytree"],
            "n_estimators": int(best_params["n_estimators"]),
            "min_child_weight": best_params["min_child_weight"],
            "random_state": 42,
        }

        model = xgb.XGBClassifier(
            **params,
        )
    else:
        model = xgb.XGBClassifier(
            objective=objective,
            num_class=num_class,
            eval_metric=eval_metric,
        )

    model.fit(
        train_X,
        train_y,
        eval_set=[(test_X, test_y)],
        verbose=False,
        sample_weight=w_train,
    )

    test_preds = model.predict(test_X)
    # calculate f1 score
    f1 = f1_score(test_y, test_preds, average="weighted")
    print("f1 score on validation set:", f1)
    print("classification report:")
    print(classification_report(test_y, test_preds, target_names=classes))
    return model


if __name__ == "__main__":
    config_file = get_args().config
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_paths = config["dataset_path"]
    output_path = config["output_path"]
    os.makedirs(output_path, exist_ok=True)

    model_name = config.get("model_name", "qc_xgb_model.pkl")
    egg_classifier_name = config.get("egg_classifier_name", "egg_xgb_model.json")
    qc_classifier_name = config.get("qc_classifier_name", "qc_xgb_model.json")

    optimize_hyperparameters = config.get("optimize_hyperparameters", True)
    mask_only = config.get("mask_only", False)
    train_egg_detector = config.get("train_egg_detector", False)
    rerun_feature_extraction = config.get("rerun_feature_extraction", True)
    n_points = config.get("n_points", 10)
    n_iter = config.get("n_iter", 50)
    test_split_ratio = config.get("test_split_ratio", 0.2)
    random_state = config.get("random_state", 42)
    pixelsizes = config.get("pixelsizes", None)

    # keep a copy of the config next to the trained model
    shutil.copy(config_file, output_path)

    # several datasets can be combined into one training set
    dataset_paths = (
        [dataset_paths] if isinstance(dataset_paths, str) else list(dataset_paths)
    )
    datasets = [
        load_dataset(path, mask_only, pixelsizes, rerun_feature_extraction)
        for path in dataset_paths
    ]
    annotations_df = pd.concat([a for a, _ in datasets], ignore_index=True)
    features_df = pd.concat([f for _, f in datasets], ignore_index=True)
    for path, (dataset_annotations_df, _) in zip(dataset_paths, datasets):
        print(f"{path}: {len(dataset_annotations_df)} samples")

    print(f"features df shape: {features_df.shape}")

    if train_egg_detector:
        eggs_annotations_df = annotations_df.copy()
        eggs_annotations_df.loc[eggs_annotations_df["Class"] != "egg", "Class"] = (
            "non_egg"
        )

        qc_annotations_df = annotations_df.copy()
        qc_annotations_df["Class"] = qc_annotations_df["Class"].replace(
            {"good": "worm", "bad": "worm", "unusable": "error"}
        )
        egg_indices = eggs_annotations_df[eggs_annotations_df["Class"] == "egg"].index
        qc_annotations_df = qc_annotations_df.drop(index=egg_indices).reset_index(
            drop=True
        )
        qc_features_df = features_df.drop(index=egg_indices).reset_index(drop=True)

        print(f"Feature df shape: {features_df.shape}")

        # first, train the egg vs non-egg model
        labels = eggs_annotations_df["Class"].values
        egg_classes = np.unique(labels).tolist()
        print(f"Classes for egg model: {egg_classes}")
        class_to_int = {cls: i for i, cls in enumerate(egg_classes)}
        labels = np.array([class_to_int[cls] for cls in labels])
        train_X, test_X, train_y, test_y = train_test_split(
            features_df,
            labels,
            test_size=test_split_ratio,
            random_state=random_state,
            stratify=labels,
        )
        print(f"TrainX shape: {train_X.shape}, TestX shape: {test_X.shape}")
        print(f"train y shape: {train_y.shape}, test y shape: {test_y.shape}")

        egg_model = train_xgb_model(
            train_X,
            train_y,
            test_X,
            test_y,
            egg_classes,
            optimize_hyperparameters,
            n_points=n_points,
            n_iter=n_iter,
        )

        # now, train the qc model
        labels = qc_annotations_df["Class"].values
        qc_classes = np.unique(labels).tolist()
        print(f"QC classes: {qc_classes}")
        class_to_int = {cls: i for i, cls in enumerate(qc_classes)}
        labels = np.array([class_to_int[cls] for cls in labels])
        train_X, test_X, train_y, test_y = train_test_split(
            qc_features_df,
            labels,
            test_size=test_split_ratio,
            random_state=random_state,
            stratify=labels,
        )
        qc_model = train_xgb_model(
            train_X,
            train_y,
            test_X,
            test_y,
            qc_classes,
            optimize_hyperparameters,
            n_points=n_points,
            n_iter=n_iter,
        )

        egg_model_path = os.path.join(output_path, egg_classifier_name)
        egg_model.save_model(egg_model_path)
        qc_model_path = os.path.join(output_path, qc_classifier_name)
        qc_model.save_model(qc_model_path)

        to_save = {
            "egg_model_path": egg_classifier_name,
            "qc_model_path": qc_classifier_name,
            "egg_classes": egg_classes,
            "qc_classes": qc_classes,
            "mask_only": mask_only,
            "pixelsizes": pixelsizes,
        }
        dump(to_save, os.path.join(output_path, model_name))
    else:
        annotations_df["Class"] = annotations_df["Class"].replace(
            {"good": "worm", "bad": "worm", "unusable": "error"}
        )

        labels = annotations_df["Class"].values
        classes = np.unique(labels).tolist()
        print(f"QC classes: {classes}")
        class_to_int = {cls: i for i, cls in enumerate(classes)}
        labels = np.array([class_to_int[cls] for cls in labels])
        train_X, test_X, train_y, test_y = train_test_split(
            features_df,
            labels,
            test_size=test_split_ratio,
            random_state=random_state,
            stratify=labels,
        )
        qc_model = train_xgb_model(
            train_X,
            train_y,
            test_X,
            test_y,
            classes,
            optimize_hyperparameters,
            n_points=n_points,
            n_iter=n_iter,
        )

        qc_model_path = os.path.join(output_path, qc_classifier_name)
        qc_model.save_model(qc_model_path)

        to_save = {
            "qc_model_path": qc_classifier_name,
            "qc_classes": classes,
            "mask_only": mask_only,
            "pixelsizes": pixelsizes,
        }
        dump(to_save, os.path.join(output_path, model_name))
