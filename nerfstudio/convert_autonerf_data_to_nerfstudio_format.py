###############################################################################
# Code written by Pierre Marza (pierre.marza@insa-lyon.fr)                    #
###############################################################################

import argparse
import json
import math
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def copying_frames(
    semantic_nerf_datapath, out_folder, in_folder, ext, spec, nb_max_policy_steps
):
    folder = os.path.join(nerfstudio_datapath, out_folder)
    if not os.path.exists(folder):
        os.makedirs(folder)

        print(f"Copying {in_folder} frames...")
        frames = os.listdir(os.path.join(semantic_nerf_datapath, in_folder))
        frames = [frame for frame in frames if ext in frame and spec in frame]
        frames.sort(key=lambda x: int(x.split("_")[-1].replace(f".{ext}", "")))
        frames = [os.path.join(semantic_nerf_datapath, in_folder, f) for f in frames]

        nb_copied = 0
        for im_id, frame in tqdm(enumerate(frames)):
            frame_id = int(frame.split("_")[-1].replace(".png", ""))
            if nb_max_policy_steps != -1 and frame_id >= nb_max_policy_steps:
                break

            im_id = str(im_id + 1)
            for _ in range(5 - len(im_id)):
                im_id = "0" + im_id
            im_name = f"frame_{im_id}.{ext}"
            copy_cmd = f"cp {frame} {os.path.join(folder, im_name)}"
            os.system(copy_cmd)
            nb_copied += 1

        return nb_copied
    else:
        print(f"{out_folder} folder already exists. Not copying frames...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic_nerf_datapath", type=str)
    parser.add_argument("--nerfstudio_datapath", type=str)
    parser.add_argument("--nb_max_policy_steps", type=int)
    parser.add_argument("--gt_sem", type=int, choices=[0, 1])
    args = parser.parse_args()
    semantic_nerf_datapath = args.semantic_nerf_datapath
    nerfstudio_datapath = args.nerfstudio_datapath
    nb_max_policy_steps = args.nb_max_policy_steps
    gt_sem = args.gt_sem

    # RGB
    nb_copied_rgb = copying_frames(
        semantic_nerf_datapath=semantic_nerf_datapath,
        out_folder="images",
        in_folder="rgb",
        ext="png",
        spec="rgb",
        nb_max_policy_steps=nb_max_policy_steps,
    )

    # Sem
    if gt_sem == 0:
        in_folder_sem = "semantic_class"
    else:
        in_folder_sem = "semantic_class_gt"
    nb_copied_sem = copying_frames(
        semantic_nerf_datapath=semantic_nerf_datapath,
        out_folder="semantic_class_gt",
        in_folder=in_folder_sem,
        ext="png",
        spec="semantic_class",
        nb_max_policy_steps=nb_max_policy_steps,
    )
    assert nb_copied_rgb == nb_copied_sem

    # Creating transforms dict
    transforms = {
        "fl_x": 320.0,  # focal length x
        "fl_y": 240.0,  # focal length y
        "cx": 319.5,  # principal point x
        "cy": 239.5,  # principal point y
        "w": 640,  # image width
        "h": 480,  # image height
        "camera_model": "OPENCV",  # camera model type
        "k1": 0.0,  # first radial distorial parameter
        "k2": 0.0,  # second radial distorial parameter
        "k3": 0.0,  # third radial distorial parameter
        "k4": 0.0,  # fourth radial distorial parameter
        "p1": 0.0,  # first tangential distortion parameter
        "p2": 0.0,  # second tangential distortion parameter
        "frames": [],  # extrinsics parameters
    }

    # Adding extrinsics parameters
    auto_nerf_transforms = np.loadtxt(
        os.path.join(semantic_nerf_datapath, "traj_w_c.txt")
    ).reshape(-1, 4, 4)
    for i in range(nb_copied_rgb):
        im_id = str(i + 1)
        for _ in range(5 - len(im_id)):
            im_id = "0" + im_id
        file_path = f"images/frame_{im_id}.png"
        gt_sem_file_path = f"semantic_class_gt/frame_{im_id}.png"
        gt_depth_file_path = f"depth/frame_{im_id}.png"
        transform_matrix = auto_nerf_transforms[i]

        # Convert from OpenCV to OpenGL convention -> 180° rotation around x axis
        rot_pi = R.from_euler("x", math.pi, degrees=False).as_matrix()
        transform_matrix[:3, :3] = transform_matrix[:3, :3] @ rot_pi

        transform_matrix = transform_matrix.tolist()
        ext_param = {
            "file_path": file_path,
            "gt_sem_file_path": gt_sem_file_path,
            "depth_file_path": gt_depth_file_path,
            "transform_matrix": transform_matrix,
        }
        transforms["frames"].append(ext_param)

        # Stopping when reaching nb max frames
        if nb_max_policy_steps != -1 and i == nb_max_policy_steps - 1:
            break

    # Saving json file
    with open(os.path.join(nerfstudio_datapath, "transforms.json"), "w") as outfile:
        json.dump(transforms, outfile)
