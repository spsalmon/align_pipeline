import os

import numpy as np
import pandas as pd
import torch
from align_toolbox.deep_learning.deep_learning_tools import (
    load_segmentation_model_from_checkpoint,
)
from align_toolbox.deep_learning.utils.augmentation import (
    get_prediction_augmentation_from_model,
)
from align_toolbox.deep_learning.utils.dataset import SegmentationPredictionDataset
from align_toolbox.foundation import image_handling
from align_toolbox.foundation.image_handling import read_tiff_file
from joblib import Parallel, delayed
from sklearn import metrics
from tifffile import imwrite
from torch.utils.data import DataLoader
from tqdm import tqdm


def reshape_images_to_original_shape(images, original_shapes, padded_or_cropped="pad"):
    reshaped_images = []
    for image, original_shape in zip(images, original_shapes):
        if padded_or_cropped == "pad":
            reshaped_image = image_handling.crop_to_dim_equally(
                image, original_shape[-2], original_shape[-1]
            )
        elif padded_or_cropped == "crop":
            reshaped_image = image_handling.pad_to_dim_equally(
                image, original_shape[-2], original_shape[-1]
            )
        reshaped_images.append(reshaped_image)
    return reshaped_images


def save_prediction(prediction, image_path, output_dir):
    path = os.path.join(output_dir, os.path.basename(image_path))
    imwrite(path, prediction, compression="zlib")


def process_image(ground_truth_path, pred_path, image_path):
    try:
        ground_truth = read_tiff_file(ground_truth_path)
        prediction = read_tiff_file(pred_path)

        ground_truth = ground_truth.astype(np.uint8)
        prediction = prediction.astype(np.uint8)

        f1 = metrics.f1_score(ground_truth.flatten(), prediction.flatten())
        iou = metrics.jaccard_score(ground_truth.flatten(), prediction.flatten())

        return {
            "image": image_path,
            "mask": ground_truth_path,
            "pred": pred_path,
            "f1": f1,
            "iou": iou,
        }
    except Exception as e:
        print(f"Error processing {ground_truth_path} : {e}")
        return None


device = "cuda" if torch.cuda.is_available() else "cpu"

model_collection_dir = "/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/models/paper/body_bf"

models = os.listdir(model_collection_dir)
model_directories = [
    m for m in models if os.path.isdir(os.path.join(model_collection_dir, m))
]
# remove the directories that do not contain a model
model_directories = [
    m
    for m in model_directories
    if os.path.exists(os.path.join(model_collection_dir, m, "best_light.ckpt"))
    and os.path.exists(os.path.join(model_collection_dir, m, "database_backup"))
]


def process_model(model_dir):
    model_path = os.path.join(model_dir, "best_light.ckpt")

    database_backup = os.path.join(model_dir, "database_backup")
    test_dataframe_path = [
        f for f in os.listdir(database_backup) if f.startswith("test_dataframe")
    ][0]
    test_dataframe_path = os.path.join(database_backup, test_dataframe_path)

    output_path = os.path.join(model_dir, "test_set_predictions")
    os.makedirs(output_path, exist_ok=True)

    model = load_segmentation_model_from_checkpoint(model_path).to(device)

    preprocessing_fn = get_prediction_augmentation_from_model(model)

    # Create the dataloader
    batch_size = 8
    df = pd.read_csv(test_dataframe_path)
    image_paths = df["image"].values

    # Only process images that haven't been processed yet
    image_paths_to_segment = [
        img
        for img in image_paths
        if not os.path.exists(os.path.join(output_path, os.path.basename(img)))
    ]

    dataset = SegmentationPredictionDataset(
        image_paths_to_segment, [1], preprocessing_fn
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    total_images = len(dataset)

    model.eval()

    with torch.no_grad():
        # Create progress bar
        pbar = tqdm(
            total=total_images,
            desc="Processing images",
            unit="img",
            dynamic_ncols=True,  # Adapts to terminal width
        )

        for i, batch in enumerate(dataloader):
            img_paths, images, image_shapes = batch
            images = torch.from_numpy(images)
            images = images.to(device)
            predictions = model(images)
            predictions = predictions.cpu().numpy()
            predictions = np.squeeze(predictions) > 0.5
            predictions = predictions.astype(np.uint8)

            predictions = reshape_images_to_original_shape(
                predictions, image_shapes, padded_or_cropped="pad"
            )

            Parallel(n_jobs=16, prefer="threads")(
                delayed(save_prediction)(prediction, image_path, output_path)
                for prediction, image_path in zip(predictions, img_paths)
            )

            # Update progress bar by batch size
            pbar.update(len(img_paths))
        pbar.close()

    ground_truth_mask_paths = df["mask"].values
    prediction_mask_paths = [
        os.path.join(output_path, os.path.basename(img_path))
        for img_path in image_paths
    ]

    files_df = pd.DataFrame(
        {
            "image": image_paths,
            "ground_truth": ground_truth_mask_paths,
            "prediction": prediction_mask_paths,
        }
    )

    files_df_sorted = files_df.sort_values(by="image")
    sorted_image_paths = files_df_sorted["image"].values
    sorted_ground_truth_paths = files_df_sorted["ground_truth"].values
    sorted_prediction_paths = files_df_sorted["prediction"].values

    rows = Parallel(n_jobs=-1)(
        delayed(process_image)(gt, pred, img)
        for gt, pred, img in zip(
            sorted_ground_truth_paths, sorted_prediction_paths, sorted_image_paths
        )
    )

    # Filter out None results
    rows = [row for row in rows if row is not None]

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(model_dir, "test_f1_iou_scores.csv"), index=False)

    report_row = {
        "model": os.path.basename(model_dir),
        "mean_f1": results_df["f1"].mean(),
        "std_f1": results_df["f1"].std(),
        "ste_f1": results_df["f1"].std() / np.sqrt(len(results_df)),
        "mean_iou": results_df["iou"].mean(),
        "std_iou": results_df["iou"].std(),
        "ste_iou": results_df["iou"].std() / np.sqrt(len(results_df)),
        "mean_f1_ti2": results_df[results_df["image"].str.contains("ti2")]["f1"].mean(),
        "std_f1_ti2": results_df[results_df["image"].str.contains("ti2")]["f1"].std(),
        "ste_f1_ti2": results_df[results_df["image"].str.contains("ti2")]["f1"].std()
        / np.sqrt(len(results_df[results_df["image"].str.contains("ti2")])),
        "mean_iou_ti2": results_df[results_df["image"].str.contains("ti2")][
            "iou"
        ].mean(),
        "std_iou_ti2": results_df[results_df["image"].str.contains("ti2")]["iou"].std(),
        "ste_iou_ti2": results_df[results_df["image"].str.contains("ti2")]["iou"].std()
        / np.sqrt(len(results_df[results_df["image"].str.contains("ti2")])),
        "mean_f1_squid": results_df[results_df["image"].str.contains("squid")][
            "f1"
        ].mean(),
        "std_f1_squid": results_df[results_df["image"].str.contains("squid")][
            "f1"
        ].std(),
        "ste_f1_squid": results_df[results_df["image"].str.contains("squid")][
            "f1"
        ].std()
        / np.sqrt(len(results_df[results_df["image"].str.contains("squid")])),
        "mean_iou_squid": results_df[results_df["image"].str.contains("squid")][
            "iou"
        ].mean(),
        "std_iou_squid": results_df[results_df["image"].str.contains("squid")][
            "iou"
        ].std(),
        "ste_iou_squid": results_df[results_df["image"].str.contains("squid")][
            "iou"
        ].std()
        / np.sqrt(len(results_df[results_df["image"].str.contains("squid")])),
    }

    print(report_row)
    return report_row


report_rows = []
for model_dir in model_directories:
    full_model_dir = os.path.join(model_collection_dir, model_dir)
    print(f"Processing model in {full_model_dir}")
    report_row = process_model(full_model_dir)
    report_rows.append(report_row)

report_df = pd.DataFrame(report_rows)
report_df.to_csv(
    os.path.join(model_collection_dir, "model_performance_report.csv"), index=False
)
