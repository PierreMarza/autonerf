################################################################
# Code adapted from https://github.com/Harry-Zhi/semantic_nerf #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)          #
################################################################

import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import torch
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_auto_sem_mask_from_path(
    filepath: Path,
) -> torch.Tensor:
    """Loads AutoNeRF semantic GT masks.

    Args:
        filepath: Path to sem mask.

    Returns:
        Sem mask torch tensor with shape [width, height, 1].
    """
    semantic = cv2.imread(str(filepath.absolute()), cv2.IMREAD_UNCHANGED)
    return torch.from_numpy(semantic)


def nanmean(data, **args):
    # This makes it ignore the first 'background' class
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)


def get_sem_metrics(outputs, batch):
    """Computes semantic metrics.

    Args:
        outputs: NeRF model outputs.
        batch: Ground-truth values.

    Returns:
        List of evaluation metrics.
    """
    predicted_labels = (
        torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
        .flatten()
        .cpu()
        .numpy()
    )
    true_labels = batch["semantics"].flatten().cpu().numpy()
    number_classes = 16
    conf_mat = confusion_matrix(
        true_labels, predicted_labels, labels=list(range(number_classes))
    )
    norm_conf_mat = np.transpose(
        np.transpose(conf_mat) / conf_mat.astype(float).sum(axis=1)
    )

    missing_class_mask = np.isnan(
        norm_conf_mat.sum(1)
    )  # missing class will have NaN at corresponding class
    exsiting_class_mask = ~missing_class_mask

    class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
    total_accuracy = np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat)
    ious = np.zeros(number_classes)
    for class_id in range(number_classes):
        ious[class_id] = conf_mat[class_id, class_id] / (
            np.sum(conf_mat[class_id, :])
            + np.sum(conf_mat[:, class_id])
            - conf_mat[class_id, class_id]
        )
    miou = nanmean(ious)
    miou_valid_class = np.mean(ious[exsiting_class_mask])

    return class_average_accuracy, total_accuracy, miou, miou_valid_class


def calculate_depth_metrics(depth_trgt, depth_pred):
    """Computes 2d metrics between two depth maps

    Args:
        depth_pred: mxn np.array containing prediction
        depth_trgt: mxn np.array containing ground truth
    Returns:
        Dict of metrics
    """
    mask1 = depth_pred > 0  # ignore values where prediction is 0 (% complete)
    mask = (depth_trgt < 10) * (depth_trgt > 0) * mask1

    depth_pred = depth_pred[mask]
    depth_trgt = depth_trgt[mask]
    abs_diff = np.abs(depth_pred - depth_trgt)
    abs_rel = abs_diff / depth_trgt
    sq_diff = abs_diff**2
    sq_rel = sq_diff / depth_trgt
    sq_log_diff = (np.log(depth_pred) - np.log(depth_trgt)) ** 2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r1 = (thresh < 1.25).astype("float")
    r2 = (thresh < 1.25**2).astype("float")
    r3 = (thresh < 1.25**3).astype("float")

    metrics = {}
    metrics["AbsRel"] = np.mean(abs_rel)
    metrics["AbsDiff"] = np.mean(abs_diff)
    metrics["SqRel"] = np.mean(sq_rel)
    metrics["RMSE"] = np.sqrt(np.mean(sq_diff))
    metrics["LogRMSE"] = np.sqrt(np.mean(sq_log_diff))
    metrics["r1"] = np.mean(r1)
    metrics["r2"] = np.mean(r2)
    metrics["r3"] = np.mean(r3)
    metrics["complete"] = np.mean(mask1.astype("float"))

    return metrics


def apply_sem_colormap(
    sem_mask,
):
    """Applies a color palette to a semantic mask

    Args:
        sem_mask: semantic mask.

    Returns:
        Colored semantic mask.
    """
    color_palette = torch.Tensor(
        [
            [0.6, 0.6, 0.6],
            [0.95, 0.95, 0.95],
            [0.96, 0.36, 0.26],
            [0.12156862745098039, 0.47058823529411764, 0.7058823529411765],
            [0.9400000000000001, 0.7818, 0.66],
            [0.9400000000000001, 0.8868, 0.66],
            [0.8882000000000001, 0.9400000000000001, 0.66],
            [0.7832000000000001, 0.9400000000000001, 0.66],
            [0.6782000000000001, 0.9400000000000001, 0.66],
            [0.66, 0.9400000000000001, 0.7468000000000001],
            [0.66, 0.9400000000000001, 0.8518000000000001],
            [0.66, 0.9232, 0.9400000000000001],
            [0.66, 0.8182, 0.9400000000000001],
            [0.66, 0.7132, 0.9400000000000001],
            [0.7117999999999999, 0.66, 0.9400000000000001],
            [0.8168, 0.66, 0.9400000000000001],
            [0.9218, 0.66, 0.9400000000000001],
            [0.9400000000000001, 0.66, 0.8531999999999998],
            [0.9400000000000001, 0.66, 0.748199999999999],
        ]
    )
    return color_palette[sem_mask]
