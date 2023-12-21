###############################################################################
# Code written by Pierre Marza (pierre.marza@insa-lyon.fr)                    #
###############################################################################

import cv2
import math
import numpy as np
import os
from scipy.spatial.transform import Rotation as R


def checking_arr_in_list(arr, list, args):
    for arr_ in list:
        if np.linalg.norm(arr[:1] - arr_[:1]) < 0.01 and abs(
            arr[2] - arr_[2]
        ) < math.radians(1):
            return True
    return False


def saving_autonerf_data(
    infos,
    autonerf_data,
    num_scenes,
    args,
    frame_ids,
    autonerf_dataset_path,
    scene2traj_id,
    pose_sets,
    floor_levels,
    sem_gt,
    instance_sem_gt,
    wait_env,
):
    assert wait_env.shape[0] == 1
    if wait_env.item() == 0:
        # Camera pose
        autonerf_sim_locations = np.asarray(
            [infos[env_idx]["sim_location"] for env_idx in range(num_scenes)]
        )

        for i in range(num_scenes):
            # Check whether pose already in set (to avoid duplicates)
            if (
                checking_arr_in_list(autonerf_sim_locations[i], pose_sets[i], args)
                is False
            ):
                pose_sets[i].append(autonerf_sim_locations[i])

                # Rotation
                rot = R.from_euler("y", autonerf_sim_locations[i, 2], degrees=False)
                rot_matrix = rot.as_matrix()

                rot_pi = R.from_euler("x", math.pi, degrees=False).as_matrix()
                rot_matrix = rot_matrix @ rot_pi

                # Translation
                trans = np.array(
                    [
                        -autonerf_sim_locations[i, 1].item(),
                        args.camera_height,
                        -autonerf_sim_locations[i, 0].item(),
                    ]
                )
                trans = np.expand_dims(trans, axis=-1)

                c2w = np.concatenate([rot_matrix, trans], axis=1)
                c2w = np.concatenate([c2w, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

                # Open traj_w_c file
                scene_name = infos[i]["scene_name"]
                traj_id = str(scene2traj_id[scene_name])
                floor_level = floor_levels[i]
                scene_path = os.path.join(
                    autonerf_dataset_path,
                    scene_name,
                    f"floor_level_{floor_level}",
                    traj_id,
                )

                if not os.path.exists(scene_path):
                    os.makedirs(scene_path)

                poses_file = os.path.join(scene_path, "traj_w_c.txt")
                f = open(poses_file, "a")

                # Writing poses to file
                poses_line = ""
                for r in range(c2w.shape[0]):
                    for c in range(c2w.shape[1]):
                        poses_line += str(c2w[r, c])
                        if r < c2w.shape[0] - 1 or c < c2w.shape[0] - 1:
                            poses_line += " "
                poses_line += "\n"
                f.write(poses_line)
                f.close()

                # RGB
                autonerf_rgb = autonerf_data[i, :, :, :3].astype(np.uint8)
                autonerf_rgb = autonerf_rgb[..., ::-1]

                # Depth
                autonerf_depth = autonerf_data[i, :, :, 3:4]
                # depth -> meters
                autonerf_depth = (
                    args.min_depth + (args.max_depth - args.min_depth) * autonerf_depth
                )
                # meters -> mmeters
                autonerf_depth *= 1000
                # to uint16
                autonerf_depth = autonerf_depth.astype(np.uint16)

                # Semantics from MaskRCNN
                autonerf_semantic = autonerf_data[i, :, :, 4:]
                autonerf_semantic_ = np.zeros(
                    (autonerf_semantic.shape[0], autonerf_semantic.shape[1])
                )
                indices_to_update = autonerf_semantic.max(axis=-1) > 0
                if indices_to_update.sum() > 0:
                    autonerf_semantic_[indices_to_update] = (
                        autonerf_semantic[indices_to_update].argmax(axis=-1) + 1
                    )
                autonerf_semantic_ = autonerf_semantic_.astype(np.uint8)

                # Semantics from gt
                autonerf_semantic_gt = sem_gt[i].astype(np.uint8)
                autonerf_instance_sem_gt = instance_sem_gt[i].astype(np.uint8)

                # Saving rgb, depth and semantic
                rgb_path = os.path.join(scene_path, "rgb")
                if not os.path.exists(rgb_path):
                    os.makedirs(rgb_path)
                depth_path = os.path.join(scene_path, "depth")
                if not os.path.exists(depth_path):
                    os.makedirs(depth_path)
                sem_path = os.path.join(scene_path, "semantic_class")
                if not os.path.exists(sem_path):
                    os.makedirs(sem_path)
                sem_path_gt = os.path.join(scene_path, "semantic_class_gt")
                if not os.path.exists(sem_path_gt):
                    os.makedirs(sem_path_gt)
                instance_sem_path_gt = os.path.join(scene_path, "instance_sem_gt")
                if not os.path.exists(instance_sem_path_gt):
                    os.makedirs(instance_sem_path_gt)

                frame_id = int(frame_ids[i].item())
                cv2.imwrite(
                    os.path.join(rgb_path, f"rgb_{frame_id}.png"),
                    autonerf_rgb,
                )
                cv2.imwrite(
                    os.path.join(depth_path, f"depth_{frame_id}.png"),
                    autonerf_depth,
                )
                cv2.imwrite(
                    os.path.join(sem_path, f"semantic_class_{frame_id}.png"),
                    autonerf_semantic_,
                )
                cv2.imwrite(
                    os.path.join(sem_path_gt, f"semantic_class_{frame_id}.png"),
                    autonerf_semantic_gt,
                )
                cv2.imwrite(
                    os.path.join(instance_sem_path_gt, f"instance_sem_{frame_id}.png"),
                    autonerf_instance_sem_gt,
                )
        frame_ids += 1
    return frame_ids
