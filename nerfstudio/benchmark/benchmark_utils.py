###############################################################################
# Code written by Pierre Marza (pierre.marza@insa-lyon.fr)                    #
###############################################################################

import json
import math
import numpy as np
from numpy import ma
import os
from pathlib import Path
import skfmm
import skimage
import torch
import torch.nn as nn
from tqdm import tqdm

from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.utils.eval_utils import eval_setup


from nerfstudio.pipelines.base_pipeline import Pipeline
from typing import Tuple
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

# Code adapted from https://github.com/nerfstudio-project/nerfstudio/blob/626441e15e8e59970ba95229e40727458bbf65a3/nerfstudio/exporter/exporter_utils.py#L79
def generate_point_cloud(
    pipeline: Pipeline,
    num_points: int = 1000000,
    use_bounding_box: bool = True,
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
):

    # pylint: disable=too-many-statements

    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    points = []
    sems = []
    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        while not progress_bar.finished:
            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_train(0)
                outputs = pipeline.model(ray_bundle)
            sem = outputs["semantics"]
            depth = outputs["depth"]
            point = ray_bundle.origins + ray_bundle.directions * depth

            if use_bounding_box:
                comp_l = torch.tensor(bounding_box_min, device=point.device)
                comp_m = torch.tensor(bounding_box_max, device=point.device)
                assert torch.all(
                    comp_l < comp_m
                ), f"Bounding box min {bounding_box_min} must be smaller than max {bounding_box_max}"
                mask = torch.all(
                    torch.concat([point > comp_l, point < comp_m], dim=-1), dim=-1
                )
                point = point[mask]
                sem = sem[mask]

            points.append(point)
            sems.append(sem)
            progress.advance(task, point.shape[0])
    points = torch.cat(points, dim=0)
    sems = torch.cat(sems, dim=0)
    return points.cpu(), sems.cpu()


def compute_occ_grid(
    nerf_model_path,
    bounds,
    meters_per_pixel,
    min_height=0.0,
    max_height=1.5,
    threshold=0.99,
):
    # Loading field
    _, pipeline, _ = eval_setup(Path(os.path.join(nerf_model_path, "config.yml")))
    field = pipeline._model.field
    spatial_distortion = field.spatial_distortion

    # Loading transforms
    f = open(os.path.join(nerf_model_path, "dataparser_transforms.json"))
    dataparser_transforms = json.load(f)
    transform = torch.Tensor(dataparser_transforms["transform"])
    scale_factor = dataparser_transforms["scale"]

    # Query points
    test_poses_min = np.zeros((3,))
    test_poses_max = np.zeros((3,))

    test_poses_min[0] = bounds[0][0]
    test_poses_min[1] = min_height
    test_poses_min[2] = bounds[0][2]

    test_poses_max[0] = bounds[1][0]
    test_poses_max[1] = max_height
    test_poses_max[2] = bounds[1][2]

    grid_dim = (test_poses_max - test_poses_min) / meters_per_pixel

    linspace_0 = torch.linspace(
        test_poses_min[0], test_poses_max[0], steps=int(grid_dim[0])
    )
    linspace_1 = torch.linspace(
        test_poses_min[1], test_poses_max[1], steps=int(grid_dim[1])
    )
    linspace_2 = torch.linspace(
        test_poses_min[2], test_poses_max[2], steps=int(grid_dim[2])
    )

    grid_query_pts = torch.meshgrid(linspace_0, linspace_1, linspace_2)

    grid_query_pts = torch.cat(
        (
            grid_query_pts[0][..., None],
            grid_query_pts[1][..., None],
            grid_query_pts[2][..., None],
        ),
        dim=3,
    )
    grid_query_pts = grid_query_pts.reshape(-1, 3)

    # Transforming poses
    grid_query_pts = torch.cat(
        [grid_query_pts, torch.ones(grid_query_pts.shape[0], 1)], dim=1
    )
    grid_query_pts = grid_query_pts.permute((1, 0))

    grid_query_pts = transform @ grid_query_pts
    grid_query_pts *= scale_factor
    grid_query_pts = grid_query_pts.permute((1, 0))

    # Spatial distorion
    assert spatial_distortion is not None
    grid_query_pts = spatial_distortion(grid_query_pts)
    grid_query_pts = (grid_query_pts + 2.0) / 4.0

    # NeRF inference
    density = []
    chunk_size = 2**15
    nb_iters = math.ceil(grid_query_pts.shape[0] / chunk_size)
    for i in tqdm(range(nb_iters)):
        with torch.no_grad():
            points = grid_query_pts[i * chunk_size : (i + 1) * chunk_size]
            points = points.cuda()

            h = field.mlp_base(points)
            density_before_activation, base_mlp_out = torch.split(
                h, [1, field.geo_feat_dim], dim=-1
            )

            # Density
            curr_density = trunc_exp(density_before_activation.to(grid_query_pts))
            density.append(curr_density)

    density = torch.cat(density, dim=0)
    density = density.reshape(int(grid_dim[0]), int(grid_dim[1]), int(grid_dim[2]))

    def occupancy_activation(alpha, distances):
        occ = 1.0 - torch.exp(-alpha * distances)
        return occ

    voxel_size = (1000 - 0.05) / 256
    density = occupancy_activation(density, voxel_size)

    # Getting binary occupancy grid
    occupancy_grid = density.detach().cpu().numpy()
    occupancy_grid = occupancy_grid > threshold
    occupancy_grid = occupancy_grid.sum(axis=1)
    occupancy_grid = occupancy_grid > 0.0

    return occupancy_grid


def compute_sem_grid(
    nerf_model_path,
    bounds,
    meters_per_pixel,
    min_height=0.0,
    max_height=1.5,
):
    # Loading pipeline
    _, pipeline, _ = eval_setup(Path(os.path.join(nerf_model_path, "config.yml")))

    pcd, sems = generate_point_cloud(
        pipeline=pipeline,
        num_points=1e7,
        use_bounding_box=True,
        bounding_box_min=(-1, -1, -1),
        bounding_box_max=(1, 1, 1),
    )

    # Loading transforms
    f = open(os.path.join(nerf_model_path, "dataparser_transforms.json"))
    dataparser_transforms = json.load(f)
    transform = torch.Tensor(dataparser_transforms["transform"])
    scale_factor = dataparser_transforms["scale"]

    # Transforming pcd
    pcd /= scale_factor

    pcd = torch.cat([torch.Tensor(pcd), torch.ones(pcd.shape[0], 1)], dim=1)
    pcd = pcd.permute((1, 0))
    transform = torch.cat([transform, torch.Tensor([[0, 0, 0, 1]])], dim=0)
    pcd = torch.inverse(transform) @ pcd
    pcd = pcd[:3].permute((1, 0)).numpy()

    # Query points
    test_poses_min = np.zeros((3,))
    test_poses_max = np.zeros((3,))

    test_poses_min[0] = bounds[0][0]
    test_poses_min[1] = min_height
    test_poses_min[2] = bounds[0][2]

    test_poses_max[0] = bounds[1][0]
    test_poses_max[1] = max_height
    test_poses_max[2] = bounds[1][2]

    grid_dim = (test_poses_max - test_poses_min) / meters_per_pixel

    voxel_indices = (pcd - test_poses_min) // meters_per_pixel
    voxel_indices = voxel_indices.astype(np.int_)

    valid_points = (
        ((voxel_indices >= 0).sum(-1) == 3)
        * (voxel_indices[:, 0] < int(grid_dim[0]))
        * (voxel_indices[:, 1] < int(grid_dim[1]))
        * (voxel_indices[:, 2] < int(grid_dim[2]))
    )
    valid_indices = voxel_indices[valid_points]

    occupancy_grid = np.zeros((int(grid_dim[0]), int(grid_dim[1]), int(grid_dim[2])))
    occupancy_grid[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = 1
    occupancy_grid = occupancy_grid.sum(axis=1) > 0

    sem_grid = np.zeros((int(grid_dim[0]), int(grid_dim[2]), 16))
    val_sems = sems[valid_points]
    for i in tqdm(range(valid_indices.shape[0])):
        label = torch.argmax(val_sems[i])
        sem_grid[valid_indices[i, 0], valid_indices[i, 2], label.item()] = 1

    return sem_grid


def map_processing(map):
    # First processing
    map = skimage.morphology.remove_small_holes(map.astype(bool), 2000)
    if map.sum().item() > 2000:
        map = skimage.morphology.remove_small_objects(map.astype(bool), 10)

    # Second processing
    selem = skimage.morphology.disk(20)
    map = skimage.morphology.binary_dilation(map.astype(bool), selem)
    map = skimage.morphology.remove_small_holes(map.astype(bool), 2000)
    map = skimage.morphology.binary_erosion(map.astype(bool), selem)
    return map


###############################################################################################################
# Code adapted from https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/envs/utils/fmm_planner.py
def get_mask(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + (
                (j + 0.5) - (size // 2 + sy)
            ) ** 2 <= step_size**2 and ((i + 0.5) - (size // 2 + sx)) ** 2 + (
                (j + 0.5) - (size // 2 + sy)
            ) ** 2 > (
                step_size - 1
            ) ** 2:
                mask[i, j] = 1

    mask[size // 2, size // 2] = 1
    return mask


def get_dist(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + (
                (j + 0.5) - (size // 2 + sy)
            ) ** 2 <= step_size**2:
                mask[i, j] = max(
                    5,
                    (
                        ((i + 0.5) - (size // 2 + sx)) ** 2
                        + ((j + 0.5) - (size // 2 + sy)) ** 2
                    )
                    ** 0.5,
                )
    return mask


class FMMPlanner:
    def __init__(self, traversible, step_size=5):
        self.step_size = step_size
        self.traversible = traversible
        self.du = int(self.step_size)
        self.fmm_dist = None

    def set_goal(self, goal, start):
        traversible = self.traversible.copy()

        traversible[start[0] - 40 : start[0] + 40, start[1] - 40 : start[1] + 40] = 1
        traversible[goal[0] - 40 : goal[0] + 40, goal[1] - 40 : goal[1] + 40] = 1

        traversible_ma = ma.masked_values(traversible * 1, 0)
        goal_x, goal_y = int(goal[0]), int(goal[1])

        traversible_ma[goal_x, goal_y] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd

        return

    def set_multi_goal(self, goal_map, start):
        traversible = self.traversible.copy()
        traversible[start[0] - 40 : start[0] + 40, start[1] - 40 : start[1] + 40] = 1
        traversible_ma = ma.masked_values(traversible * 1, 0)
        traversible_ma[goal_map == 1] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd

        return

    def get_short_term_goal(self, state):
        scale = 1.0
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask = get_mask(dx, dy, scale, self.step_size)
        dist_mask = get_dist(dx, dy, scale, self.step_size)

        state = [int(x) for x in state]

        dist = np.pad(
            self.fmm_dist,
            self.du,
            "constant",
            constant_values=self.fmm_dist.shape[0] ** 2,
        )
        subset = dist[
            state[0] : state[0] + 2 * self.du + 1, state[1] : state[1] + 2 * self.du + 1
        ]

        assert (
            subset.shape[0] == 2 * self.du + 1 and subset.shape[1] == 2 * self.du + 1
        ), f"Planning error: unexpected subset shape {subset.shape}"

        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2

        if subset[self.du, self.du] < 0.25 * 100 / 5.0:  # 25cm
            stop = True
        else:
            stop = False

        subset -= subset[self.du, self.du]
        ratio1 = subset / dist_mask
        subset[ratio1 < -1.5] = 1

        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        if subset[stg_x, stg_y] > -0.0001:
            replan = True
        else:
            replan = False

        return (
            (stg_x + state[0] - self.du) * scale,
            (stg_y + state[1] - self.du) * scale,
            replan,
            stop,
        )


###############################################################################################################


def convert_points_to_topdown(bounds, points, meters_per_pixel):
    points_topdown = []
    for point in points:
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


def convert_pose_to_occ_grid(pose, test_poses_min, res):
    occ_grid_pose = [
        int((pose[0] - test_poses_min[0]) / res),
        int((pose[2] - test_poses_min[2]) / res),
    ]
    return occ_grid_pose


def convert_occ_grid_to_pose(occ_grid_pose, test_poses_min, res):
    pose = [
        test_poses_min[0] + occ_grid_pose[0] * res,
        test_poses_min[2] + occ_grid_pose[1] * res,
    ]
    return pose


def get_success(path, gt_map, end_topdown):
    last_p = path[-1]
    if np.linalg.norm(np.array(last_p) - np.array(end_topdown)) >= 100:
        return 0

    nb_obstacle_hits = 0
    for point in path:
        point[0], point[1] = point[1], point[0]
        for p in [
            [point[0] - 1, point[1] - 1],
            [point[0] - 1, point[1]],
            [point[0] - 1, point[1] + 1],
            [point[0], point[1] - 1],
            [point[0], point[1]],
            [point[0], point[1] + 1],
            [point[0] + 1, point[1] - 1],
            [point[0] + 1, point[1]],
            [point[0] + 1, point[1] + 1],
        ]:
            if (
                int(p[0]) < 0
                or int(p[0]) >= gt_map.shape[0]
                or int(p[1]) < 0
                or int(p[1]) >= gt_map.shape[1]
                or gt_map[int(p[0]), int(p[1])] == False
            ):
                nb_obstacle_hits += 1

    if nb_obstacle_hits <= 10:
        return 1
    else:
        return 0


def plan_and_eval(
    start_type,
    end_type,
    start,
    end,
    points,
    planning_map,
    goal_map,
    sim_topdown_map,
    bounds,
    test_poses_min,
    class_id,
    meters_per_pixel,
):
    assert start_type in ["point", "sem"]
    assert end_type in ["point", "sem"]

    if end_type == "point":
        assert goal_map is None
        assert class_id is None

    [_, end_topdown] = convert_points_to_topdown(
        bounds, [start, end], meters_per_pixel=meters_per_pixel
    )
    path_topdown = convert_points_to_topdown(
        bounds, points, meters_per_pixel=meters_per_pixel
    )
    path_topdown = np.array(path_topdown)

    start = convert_pose_to_occ_grid(start, test_poses_min, meters_per_pixel)
    end = convert_pose_to_occ_grid(end, test_poses_min, meters_per_pixel)

    fmm_planner = FMMPlanner(traversible=planning_map, step_size=5)
    if end_type == "point":
        fmm_planner.set_goal(goal=end, start=start)
    else:
        fmm_planner.set_multi_goal(goal_map=goal_map, start=start)

    stop = False
    state = start
    pred_points = []
    nb_steps = 0
    while not stop and nb_steps < 1000:
        stg_x, stg_y, _, stop = fmm_planner.get_short_term_goal(state=state)
        stg_x, stg_y = int(stg_x), int(stg_y)
        state = [stg_x, stg_y]
        pred_points.append([stg_x, stg_y])
        nb_steps += 1
    pred_points = np.array(pred_points)

    pred_points_ = []
    for p in pred_points:
        p_3d = convert_occ_grid_to_pose(p, test_poses_min, meters_per_pixel)
        p_3d = [p_3d[0], None, p_3d[1]]
        pred_points_.append(p_3d)
    planned_path_3d = pred_points_.copy()
    planned_path_3d = np.array(planned_path_3d)
    pred_points_ = convert_points_to_topdown(
        bounds, pred_points_, meters_per_pixel=meters_per_pixel
    )
    pred_points_ = np.array(pred_points_)
    succ = get_success(pred_points_, sim_topdown_map, end_topdown)

    # Converting both paths to world frame of reference to compute spl
    gt_length = 0.0
    for i in range(1, len(points)):
        gt_length += np.linalg.norm(
            np.array([points[i, 0], points[i, 2]])
            - np.array([points[i - 1, 0], points[i - 1, 2]])
        )
    planned_path_length = 0.0
    for i in range(1, len(planned_path_3d)):
        planned_path_length += np.linalg.norm(
            np.array([planned_path_3d[i, 0], planned_path_3d[i, 2]])
            - np.array([planned_path_3d[i - 1, 0], planned_path_3d[i - 1, 2]])
        )
    spl = succ * (gt_length / max(gt_length, planned_path_length))

    return succ, spl


###############################################################################################################
# Code adapted from https://github.com/salykovaa/inerf/blob/main/inerf_helpers.py
def vec2ss_matrix(vector):  # vector to skewsym. matrix
    ss_matrix = torch.zeros((3, 3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix


class camera_transf(nn.Module):
    def __init__(self, device):
        super(camera_transf, self).__init__()
        self.w = nn.Parameter(torch.normal(0.0, 1e-6, size=(3,)))
        self.v = nn.Parameter(torch.normal(0.0, 1e-6, size=(3,)))
        self.theta = nn.Parameter(torch.normal(0.0, 1e-6, size=()))
        self.device = device

    def forward(self, x):
        exp_i = torch.zeros((4, 4)).to(self.device)
        w_skewsym = vec2ss_matrix(self.w).to(self.device)

        exp_i[:3, :3] = (
            torch.eye(3).to(self.device)
            + torch.sin(self.theta) * w_skewsym
            + (1 - torch.cos(self.theta)) * torch.matmul(w_skewsym, w_skewsym)
        )

        exp_i[:3, 3] = torch.matmul(
            torch.eye(3).to(self.device) * self.theta
            + (1 - torch.cos(self.theta)) * w_skewsym
            + (self.theta - torch.sin(self.theta)) * torch.matmul(w_skewsym, w_skewsym),
            self.v,
        )

        exp_i[3, 3] = 1.0
        T_i = torch.matmul(exp_i, x)
        return T_i


to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

rot_psi = lambda psi: np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(psi), -np.sin(psi), 0],
        [0, np.sin(psi), np.cos(psi), 0],
        [0, 0, 0, 1],
    ]
)

rot_theta = lambda th: np.array(
    [
        [np.cos(th), 0, np.sin(th), 0],
        [0, 1, 0, 0],
        [-np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
)

rot_phi = lambda phi: np.array(
    [
        [np.cos(phi), -np.sin(phi), 0, 0],
        [np.sin(phi), np.cos(phi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)

trans_t_x = lambda t: np.array([[1, 0, 0, t], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

trans_t_y = lambda t: np.array([[1, 0, 0, 0], [0, 1, 0, t], [0, 0, 1, 0], [0, 0, 0, 1]])

trans_t_z = lambda t: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]])
###############################################################################################################
