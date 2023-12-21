###############################################################################
# Code written by Pierre Marza (pierre.marza@insa-lyon.fr)                    #
###############################################################################

import argparse
import json
import numpy as np
import os
import pickle
import skimage.morphology
import skimage.measure
from tqdm import tqdm

from benchmark_utils import compute_occ_grid, compute_sem_grid, plan_and_eval


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
    savedir = f"benchmark/data/gt_planning/{scene}"
    sim_topdown_map = np.load(os.path.join(savedir, "sim_topdown_map.npy"))
    with open(os.path.join(savedir, "paths.pkl"), "rb") as f:
        paths = pickle.load(f)
    with open(os.path.join(savedir, "bounds.pkl"), "rb") as f:
        bounds = pickle.load(f)

    # Creating test_poses_min and test_poses_max
    test_poses_min = np.zeros((3,))
    test_poses_max = np.zeros((3,))
    test_poses_min[0] = bounds[0][0]
    test_poses_min[2] = bounds[0][2]
    test_poses_max[0] = bounds[1][0]
    test_poses_max[2] = bounds[1][2]

    # Selecting eval samples
    paths = paths[:100]

    # Metrics dict to save as json
    metrics = {}
    metrics_file = os.path.join(nerf_model_path, "benchmark_planning.json")

    # Computing occupancy and semantic grids
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

    # Processing occupancy grid
    occupancy_grid = skimage.morphology.remove_small_objects(occupancy_grid, 300)
    selem = skimage.morphology.disk(20)
    occupancy_grid = skimage.morphology.binary_dilation(occupancy_grid, selem)

    grid_dim = occupancy_grid.shape
    succ_res = []
    spl_res = []
    succ_res_sem = []
    spl_res_sem = []
    for path_id, path in tqdm(enumerate(paths)):
        # Point to Point
        start = path["start"]
        end = path["point2point"]["end"]
        points = np.array(path["point2point"]["path"])  # GT path points
        succ, spl = plan_and_eval(
            start_type="point",
            end_type="point",
            start=start,
            end=end,
            points=points,
            planning_map=(occupancy_grid == 0.0),
            goal_map=None,
            sim_topdown_map=sim_topdown_map,
            bounds=bounds,
            test_poses_min=test_poses_min,
            class_id=None,
            meters_per_pixel=meters_per_pixel,
        )

        succ_res.append(succ)
        if not np.isnan(spl):
            spl_res.append(spl)
        else:
            print("spl is nan...")

        # Point to sem cat
        for class_id in path["point2sem"].keys():
            start = path["start"]
            end = path["point2sem"][class_id]["end"]
            points = np.array(path["point2sem"][class_id]["path"])  # GT path points

            # Processing sem goal map
            goal_map = sem_grid[..., class_id + 1] == 1
            selem_dilation = skimage.morphology.disk(25)
            goal_map = skimage.morphology.remove_small_objects(goal_map, 300)
            goal_map = skimage.morphology.binary_dilation(goal_map, selem_dilation)

            if True in np.unique(goal_map):
                succ, spl = plan_and_eval(
                    start_type="point",
                    end_type="sem",
                    start=start,
                    end=end,
                    points=points,
                    planning_map=(occupancy_grid == 0.0),
                    goal_map=goal_map,
                    sim_topdown_map=sim_topdown_map,
                    bounds=bounds,
                    test_poses_min=test_poses_min,
                    class_id=class_id,
                    meters_per_pixel=meters_per_pixel,
                )
            else:
                succ, spl = 0.0, 0.0
            succ_res_sem.append(succ)
            if not np.isnan(spl):
                spl_res_sem.append(spl)
            else:
                print("spl is nan...")

    # Saving metrics dict
    metrics["mean_pointgoal_succ"] = np.array(succ_res).mean().item()
    metrics["mean_pointgoal_spl"] = np.array(spl_res).mean().item()
    metrics["mean_objgoal_succ"] = np.array(succ_res_sem).mean().item()
    metrics["mean_objgoal_spl"] = np.array(spl_res_sem).mean().item()
    with open(metrics_file, "w") as outfile:
        json.dump(metrics, outfile)
