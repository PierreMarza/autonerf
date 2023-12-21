###############################################################################
# Code written by Pierre Marza (pierre.marza@insa-lyon.fr)                    #
###############################################################################

import argparse
import json
import numpy as np
import os
import pickle
import skimage
from sklearn.metrics import accuracy_score, precision_score, recall_score

from benchmark_utils import compute_occ_grid, compute_sem_grid, map_processing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, help="Gibson val scene")
    parser.add_argument("--nerf_model_path", type=str, help="Path to NeRF model")
    parser.add_argument(
        "--meters_per_pixel", type=float, default=0.01, help="Grid resolution"
    )
    args = parser.parse_args()

    scene = args.scene
    nerf_model_path = args.nerf_model_path
    meters_per_pixel = args.meters_per_pixel
    assert scene in ["Collierville", "Corozal", "Darden", "Markleeville", "Wiconisco"]
    assert scene in nerf_model_path

    # Loading Habitat GT
    savedir = f"benchmark/data/gt_occ_sem_grid/{scene}"
    ## Occupancy map
    sim_topdown_map = np.load(os.path.join(savedir, "sim_topdown_map.npy"))
    ## Semantic map
    sim_topdown_sem_map = np.load(os.path.join(savedir, "sim_topdown_sem_map.npy"))
    ## Scene bounds
    with open(os.path.join(savedir, "bounds.pkl"), "rb") as f:
        bounds = pickle.load(f)

    # Metrics dict to save as json
    metrics = {}
    metrics_file = os.path.join(nerf_model_path, "benchmark_bev_map.json")

    # Computing occupancy and semantic grids from NeRF
    occupancy_grid = compute_occ_grid(
        nerf_model_path,
        bounds,
        meters_per_pixel,
    )
    sem_grid = compute_sem_grid(
        nerf_model_path,
        bounds,
        meters_per_pixel,
    )

    # Processing occupancy_grid
    occupancy_grid = skimage.morphology.remove_small_objects(occupancy_grid, 300)
    occupancy_grid = (occupancy_grid == 0.0).T

    # Evaluating occupancy grid
    occ_acc = accuracy_score(sim_topdown_map.reshape(-1), occupancy_grid.reshape(-1))
    occ_precision = precision_score(
        sim_topdown_map.reshape(-1), occupancy_grid.reshape(-1)
    )
    occ_recall = recall_score(sim_topdown_map.reshape(-1), occupancy_grid.reshape(-1))
    metrics["occ_acc"] = occ_acc
    metrics["occ_precision"] = occ_precision
    metrics["occ_recall"] = occ_recall

    sem_grid = sem_grid.transpose((1, 0, 2))
    sem_accs = []
    sem_precisions = []
    sem_recalls = []
    for i in range(sim_topdown_sem_map.shape[0]):
        gt_sem_map = sim_topdown_sem_map[i]
        sem_grid_curr = sem_grid[:, :, i + 1]
        sem_grid_curr = sem_grid_curr * (occupancy_grid == 0.0)

        # Processing semantic grid
        sem_grid_curr = map_processing(sem_grid_curr)

        # Evaluating semantic grid
        sem_acc = accuracy_score(gt_sem_map.reshape(-1), sem_grid_curr.reshape(-1))
        sem_precision = precision_score(
            gt_sem_map.reshape(-1), sem_grid_curr.reshape(-1)
        )
        sem_recall = recall_score(gt_sem_map.reshape(-1), sem_grid_curr.reshape(-1))

        sem_accs.append(sem_acc)
        sem_precisions.append(sem_precision)
        sem_recalls.append(sem_recall)

    # Computing semantic metrics
    metrics["mean_sem_acc"] = np.mean(sem_accs)
    metrics["mean_sem_precision"] = np.mean(sem_precisions)
    metrics["mean_sem_recall"] = np.mean(sem_recalls)

    # Saving metrics dict
    with open(metrics_file, "w") as outfile:
        json.dump(metrics, outfile)
