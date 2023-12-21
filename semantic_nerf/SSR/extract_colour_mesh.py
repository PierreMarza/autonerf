################################################################
# Code adapted from https://github.com/Harry-Zhi/semantic_nerf #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)          #
################################################################

import argparse
from collections import defaultdict
import copy
import cv2
import numpy as np
import open3d as o3d
import os
import skimage.measure as ski_measure
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm
import yaml

from SSR.utils.image_utils import color_palette
from SSR.training import trainer
from SSR.models.model_utils import run_network
from SSR.visualisation import open3d_utils


@torch.no_grad()
def render_fn(trainer, rays, chunk):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = trainer.render_rays(rays[i : i + chunk])

        for k, v in rendered_ray_chunks.items():
            if k in ["sem_logits_fine", "rgb_fine"]:
                results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="SSR/configs/autonerf_config.yaml",
        help="config file name.",
    )
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--train_data_dir", type=str)
    parser.add_argument(
        "--near_t",
        type=float,
        default=2.0,
        help="the near bound factor to start the ray",
    )
    parser.add_argument("--grid_dim", type=int, default=256)
    args = parser.parse_args()

    # Read YAML file
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    trainer.select_gpus(config["experiment"]["gpu"])

    to8b_np = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    logits_2_label = lambda x: torch.argmax(
        torch.nn.functional.softmax(x, dim=-1), dim=-1
    )
    mesh_recon_save_dir = os.path.join(args.save_dir, "mesh_reconstruction")
    os.makedirs(mesh_recon_save_dir, exist_ok=True)

    # Load ckpt
    ckpt_path = os.path.join(args.save_dir, "checkpoints", "200000.ckpt")
    ckpt = torch.load(ckpt_path)
    num_valid_semantic_class = int(
        ckpt["network_coarse_state_dict"]["semantic_linear.1.bias"].shape[0]
    )
    ssr_trainer = trainer.SSRTrainer(config)
    ssr_trainer.num_valid_semantic_class = num_valid_semantic_class

    train_data_dir = args.train_data_dir

    # Semantic mesh
    color_map_np = color_palette * 255
    color_map_np = color_map_np.astype(np.uint8)
    valid_color_map_ = color_map_np[1:]

    # Re-mapping color palette as only a part of the sem classes are in the current scene
    train_gt_sem = os.path.join(train_data_dir, "semantic_class_gt/")
    train_gt_sem = os.listdir(train_gt_sem)
    train_gt_sem = [
        os.path.join(train_data_dir, "semantic_class_gt", f) for f in train_gt_sem
    ]
    sem_classes = set()
    for f in tqdm(train_gt_sem):
        sem_ = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        for c in np.unique(sem_):
            sem_classes.add(c)
    valid_color_map = []
    for sem_class in sem_classes:
        valid_color_map.append(valid_color_map_[sem_class])
    valid_color_map = np.array(valid_color_map)

    # Create nerf model, init optimizer
    ssr_trainer.create_ssr()
    ssr_trainer.ssr_net_coarse.load_state_dict(
        ckpt["network_coarse_state_dict"], strict=False
    )
    ssr_trainer.ssr_net_fine.load_state_dict(
        ckpt["network_fine_state_dict"], strict=False
    )
    ssr_trainer.training = False
    ssr_trainer.ssr_net_coarse.eval()
    ssr_trainer.ssr_net_fine.eval()

    # Hyperparameters
    H, W, near, far, level = (
        240,
        320,
        0.1,
        10.0,
        0.45,
    )
    ssr_trainer.enable_semantic = True
    ssr_trainer.save_dir = None
    ssr_trainer.H_scaled = H
    ssr_trainer.W_scaled = W
    ssr_trainer.near = near
    ssr_trainer.far = far

    train_poses = np.loadtxt(
        os.path.join(train_data_dir, "traj_w_c.txt"), delimiter=" "
    ).reshape(-1, 4, 4)
    train_poses = train_poses[:, :3, -1]
    train_poses_min = train_poses.min(axis=0)
    train_poses_max = train_poses.max(axis=0)

    # Defining generated mesh boundaries
    # +- 2.0m along x and z axes: the final mesh will require additional
    # manual post-processing to remove parts outside of the scene.
    # y-axis boundaries: -0.5m and +1.5m
    train_poses_min[0] -= 2.0
    train_poses_min[1] = -0.5
    train_poses_min[2] -= 2.0

    train_poses_max[0] += 2.0
    train_poses_max[1] = 1.5
    train_poses_max[2] += 2.0

    linspace_0 = torch.linspace(
        train_poses_min[0], train_poses_max[0], steps=args.grid_dim
    )
    linspace_1 = torch.linspace(
        train_poses_min[1], train_poses_max[1], steps=args.grid_dim
    )
    linspace_2 = torch.linspace(
        train_poses_min[2], train_poses_max[2], steps=args.grid_dim
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

    grid_query_pts = grid_query_pts.cuda().reshape(-1, 1, 3)  # Num_rays, 1, 3-xyz
    viewdirs = torch.zeros_like(grid_query_pts).reshape(-1, 3)

    with torch.no_grad():
        chunk = 1024
        run_MLP_fn = lambda pts: run_network(
            inputs=pts,
            viewdirs=torch.zeros_like(pts).squeeze(1),
            fn=ssr_trainer.ssr_net_fine,
            embed_fn=ssr_trainer.embed_fn,
            embeddirs_fn=ssr_trainer.embeddirs_fn,
            netchunk=int(2048 * 128),
        )

        raw = torch.cat(
            [
                run_MLP_fn(grid_query_pts[i : i + chunk]).cpu()
                for i in range(0, grid_query_pts.shape[0], chunk)
            ],
            dim=0,
        )
        alpha = raw[..., 3]  # [N]

    def occupancy_activation(alpha, distances):
        occ = 1.0 - torch.exp(-F.relu(alpha) * distances)
        # notice we apply RELU to raw sigma before computing alpha
        return occ

    voxel_size = (ssr_trainer.far - ssr_trainer.near) / ssr_trainer.N_importance
    occ = occupancy_activation(alpha, voxel_size)
    occ = occ.reshape(args.grid_dim, args.grid_dim, args.grid_dim)
    occupancy_grid = occ.detach().cpu().numpy()

    vertices, faces, vertex_normals, _ = ski_measure.marching_cubes(
        occupancy_grid, level=level, gradient_direction="ascent"
    )

    dim = occupancy_grid.shape[0]
    vertices = vertices / (dim - 1)
    vertices = train_poses_min + (train_poses_max - train_poses_min) * vertices
    mesh = trimesh.Trimesh(
        vertices=vertices, vertex_normals=vertex_normals, faces=faces
    )

    trimesh.exchange.export.export_mesh(
        mesh, os.path.join(mesh_recon_save_dir, "mesh.ply")
    )
    o3d_mesh = open3d_utils.trimesh_to_open3d(mesh)
    o3d_mesh_clean = open3d_utils.clean_mesh(
        o3d_mesh, keep_single_cluster=False, min_num_cluster=400
    )
    vertices_ = np.array(o3d_mesh_clean.vertices).reshape([-1, 3]).astype(np.float32)

    ## use normal vector method as suggested by the author, see https://github.com/bmild/nerf/issues/44
    mesh_recon_save_dir = os.path.join(mesh_recon_save_dir, "use_vertex_normal")
    os.makedirs(mesh_recon_save_dir, exist_ok=True)

    selected_mesh = o3d_mesh_clean
    rays_d = -torch.FloatTensor(
        np.asarray(selected_mesh.vertex_normals)
    )  # use negative normal directions as ray marching directions

    near = 0.1 * torch.ones_like(rays_d[:, :1])
    far = 10.0 * torch.ones_like(rays_d[:, :1])
    rays_o = torch.FloatTensor(vertices_) - rays_d * near * args.near_t
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True).float()
    rays = torch.cat([rays_o, rays_d, near, far, viewdirs], -1)

    # provide ray directions as input
    rays = rays.cuda()
    with torch.no_grad():
        chunk = 4096
        results = render_fn(ssr_trainer, rays, chunk)

        # Semantic mesh
        labels = logits_2_label(results["sem_logits_fine"]).numpy()
        vis_labels = valid_color_map[labels]
        v_colors_sem = vis_labels

        # RGB mesh
        rgbs = results["rgb_fine"].numpy()
        rgbs = to8b_np(rgbs)
        v_colors_rgb = rgbs

    o3d.io.write_triangle_mesh(
        os.path.join(mesh_recon_save_dir, "o3d_mesh_clean.ply"), o3d_mesh_clean
    )

    v_colors_sem = v_colors_sem.astype(np.uint8)
    v_colors_rgb = v_colors_rgb.astype(np.uint8)

    o3d_mesh_clean_sem = copy.deepcopy(o3d_mesh_clean)
    o3d_mesh_clean_rgb = copy.deepcopy(o3d_mesh_clean)

    o3d_mesh_clean_sem.vertex_colors = o3d.utility.Vector3dVector(v_colors_sem / 255.0)
    o3d_mesh_clean_rgb.vertex_colors = o3d.utility.Vector3dVector(v_colors_rgb / 255.0)

    # Semantic mesh
    o3d.io.write_triangle_mesh(
        os.path.join(
            mesh_recon_save_dir,
            f"semantic_mesh_dim{args.grid_dim}neart_{args.near_t}.ply",
        ),
        o3d_mesh_clean_sem,
    )

    # RGB mesh
    o3d.io.write_triangle_mesh(
        os.path.join(
            mesh_recon_save_dir,
            f"color_mesh_dim{args.grid_dim}neart_{args.near_t}.ply",
        ),
        o3d_mesh_clean_rgb,
    )


if __name__ == "__main__":
    main()
