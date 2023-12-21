###############################################################################
# Code written by Pierre Marza (pierre.marza@insa-lyon.fr)                    #
###############################################################################

import argparse
import cv2
import json
import math
import numpy as np
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import torch
from tqdm import tqdm

from benchmark_utils import (
    camera_transf,
    trans_t_z,
    trans_t_y,
    trans_t_x,
    rot_phi,
    rot_theta,
    rot_psi,
)
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.eval_utils import eval_setup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, help="Gibson val scene")
    parser.add_argument("--nerf_model_path", type=str, help="Path to NeRF model")
    parser.add_argument(
        "--meters_per_pixel", type=float, default=0.01, help="Grid resolution"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--lrate", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--N_iters", type=int, default=300, help="Number of optimization iterations"
    )
    args = parser.parse_args()

    scene = args.scene
    nerf_model_path = args.nerf_model_path
    meters_per_pixel = args.meters_per_pixel
    assert scene in ["Collierville", "Corozal", "Darden", "Markleeville", "Wiconisco"]
    assert scene in nerf_model_path

    # Optimization hyperparameters
    device = args.device
    lrate = args.lrate
    N_iters = args.N_iters

    # Camera config
    cam_config = {
        "fx": 320.0,
        "fy": 240.0,
        "cx": 319.5,
        "cy": 239.5,
        "distortion_params": torch.zeros(6),
        "height": 480,
        "width": 640,
        "camera_type": CameraType.PERSPECTIVE,
    }

    # GT pose perturbation
    gt_pose_perturbations = {
        "psi": 5.0,
        "theta": 5.0,
        "phi": 5.0,
        "t_x": 0.03,
        "t_y": 0.03,
        "t_z": 0.03,
    }

    # Metrics dict to save as json
    metrics = {}
    metrics_file = os.path.join(nerf_model_path, "benchmark_pose_refinement.json")

    _, pipeline, _ = eval_setup(Path(os.path.join(nerf_model_path, "config.yml")))
    model = pipeline._model

    # Loading transforms
    f = open(os.path.join(nerf_model_path, "dataparser_transforms.json"))
    dataparser_transforms = json.load(f)
    transform = torch.Tensor(dataparser_transforms["transform"])
    scale_factor = dataparser_transforms["scale"]

    # Loss
    img2mse = lambda x, y: torch.mean((x - y) ** 2)

    # GT camera poses
    all_gt_cam_trans = np.loadtxt(
        f"../semantic_nerf/data/test/{scene}/floor_level_0/traj_w_c.txt",
        delimiter=" ",
    ).reshape(-1, 4, 4)

    # Selecting eval samples
    images_folder = f"../semantic_nerf/data/test/{scene}/floor_level_0/rgb/"
    images = os.listdir(images_folder)
    images.sort(key=lambda x: int(x.split("_")[-1].replace(".png", "")))
    images = [os.path.join(images_folder, im) for im in images]
    images = images[:100]

    rot_errors = []
    translation_errors = []
    for img_id, img_path in tqdm(enumerate(images)):
        # Camera transform model and optimizer
        cam_transf = camera_transf(device).to(device)
        optimizer = torch.optim.Adam(
            params=cam_transf.parameters(), lr=lrate, betas=(0.9, 0.999)
        )

        # Image to localize
        image = (
            cv2.imread(img_path)[:, :, ::-1] / 255.0
        )  # change from BGR uinit 8 to RGB float
        image = torch.Tensor(image).to(device)

        # GT camera pose
        gt_cam_trans = all_gt_cam_trans[img_id]

        # Convert from OpenCV to OpenGL convention -> 180° rot around x axis
        rot_pi = R.from_euler("x", math.pi, degrees=False).as_matrix()
        gt_cam_trans[:3, :3] = gt_cam_trans[:3, :3] @ rot_pi

        start_pose = (
            trans_t_z(gt_pose_perturbations["t_z"])
            @ trans_t_y(gt_pose_perturbations["t_y"])
            @ trans_t_x(gt_pose_perturbations["t_x"])
            @ rot_phi(gt_pose_perturbations["phi"] / 180.0 * np.pi)
            @ rot_theta(gt_pose_perturbations["theta"] / 180.0 * np.pi)
            @ rot_psi(gt_pose_perturbations["psi"] / 180.0 * np.pi)
            @ gt_cam_trans
        )
        start_pose = np.expand_dims(start_pose, axis=0)

        # Code from https://github.com/salykovaa/inerf/blob/main/run.py
        # calculate angles and translation of the observed image's pose
        psi_ref = np.arctan2(gt_cam_trans[2, 1], gt_cam_trans[2, 2]) * 180 / np.pi
        theta_ref = (
            np.arctan2(
                -gt_cam_trans[2, 0],
                np.sqrt(gt_cam_trans[2, 1] ** 2 + gt_cam_trans[2, 2] ** 2),
            )
            * 180
            / np.pi
        )
        phi_ref = np.arctan2(gt_cam_trans[1, 0], gt_cam_trans[0, 0]) * 180 / np.pi
        translation_ref = gt_cam_trans[:3, 3]
        start_pose = torch.Tensor(start_pose).to(device)
        gt_cam_trans = torch.Tensor(gt_cam_trans).to(device)

        # Optimization
        for i in range(N_iters):
            # Predict camera pose
            pred_cam_trans = cam_transf(start_pose)

            # Transforming poses
            pred_cam_trans = transform.to(device) @ pred_cam_trans
            pred_cam_trans[:, :3, 3] *= scale_factor

            # Generate rays from camera pose
            camera_optimizer_config = CameraOptimizerConfig()
            camera_optimizer = CameraOptimizer(
                config=camera_optimizer_config, num_cameras=1, device="cpu"
            )

            cameras = Cameras(
                fx=cam_config["fx"],
                fy=cam_config["fy"],
                cx=cam_config["cx"],
                cy=cam_config["cy"],
                distortion_params=cam_config["distortion_params"],
                height=cam_config["height"],
                width=cam_config["width"],
                camera_to_worlds=pred_cam_trans.to("cpu"),
                camera_type=cam_config["camera_type"],
            )

            ray_generator = RayGenerator(
                cameras,
                camera_optimizer,
            )

            # Sampling rays
            nb_rays = 4096
            ray_indices = torch.zeros((nb_rays, 3)).long().to(device)
            ray_indices[:, 1] = torch.randint(0, 480, (nb_rays,))
            ray_indices[:, 2] = torch.randint(0, 640, (nb_rays,))
            ray_bundle = ray_generator(ray_indices)
            ray_bundle.nears = torch.ones(nb_rays, 1).to(device) * 0.05
            ray_bundle.fars = torch.ones(nb_rays, 1).to(device) * 1000
            ray_bundle = ray_bundle.to(device)

            # Volumetric rendering
            output_dict = model.get_outputs(ray_bundle)
            rgb = output_dict["rgb"]

            # Compute loss
            sampled_gt_rgb = image[ray_indices[:, 1], ray_indices[:, 2]]
            loss = img2mse(rgb, sampled_gt_rgb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            new_lrate = lrate * (0.8 ** ((i + 1) / 100))
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lrate

        # Code from https://github.com/salykovaa/inerf/blob/main/run.py
        with torch.no_grad():
            pred_cam_trans = cam_transf(start_pose)
            pose_dummy = pred_cam_trans[0].cpu().detach().numpy()
            # calculate angles and translation of the optimized pose
            psi = np.arctan2(pose_dummy[2, 1], pose_dummy[2, 2]) * 180 / np.pi
            theta = (
                np.arctan2(
                    -pose_dummy[2, 0],
                    np.sqrt(pose_dummy[2, 1] ** 2 + pose_dummy[2, 2] ** 2),
                )
                * 180
                / np.pi
            )
            phi = np.arctan2(pose_dummy[1, 0], pose_dummy[0, 0]) * 180 / np.pi
            translation = pose_dummy[:3, 3]

            # calculate error between optimized and observed pose
            psi_error = (psi_ref - psi) % 360
            psi_error = min(360 - psi_error, psi_error)
            theta_error = (theta_ref - theta) % 360
            theta_error = min(360 - theta_error, theta_error)
            phi_error = (phi_ref - phi) % 360
            phi_error = min(360 - phi_error, phi_error)

            rot_error = psi_error + theta_error + phi_error
            translation_error = np.linalg.norm(translation_ref - translation)

            rot_errors.append(rot_error)
            translation_errors.append(translation_error)

    # Computing metrics
    rot_errors = np.array(rot_errors)
    translation_errors = np.array(translation_errors)
    rot_converged_samples = rot_errors < 3
    translation_converged_samples = translation_errors < 0.02

    conv_rate = (
        rot_converged_samples * translation_converged_samples
    ).sum().item() / len(translation_errors)
    rot_error = (
        rot_errors[(rot_converged_samples * translation_converged_samples)]
        .mean()
        .item()
    )
    trans_error = (
        translation_errors[(rot_converged_samples * translation_converged_samples)]
        .mean()
        .item()
    )

    metrics["conv_rate"] = conv_rate
    metrics["rot_error"] = rot_error
    metrics["trans_error"] = trans_error

    # Saving metrics dict
    with open(metrics_file, "w") as outfile:
        json.dump(metrics, outfile)
