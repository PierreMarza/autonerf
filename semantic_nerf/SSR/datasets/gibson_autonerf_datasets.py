################################################################
# Code adapted from https://github.com/Harry-Zhi/semantic_nerf #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)          #
################################################################

import cv2
import glob
import numpy as np
import os
from torch.utils.data import Dataset

from SSR.utils.image_utils import color_palette


class GibsonAutoNeRFDatasetCache(Dataset):
    def __init__(
        self, train_data_dir, test_data_dir, img_h=None, img_w=None, use_GT_sem=False
    ):
        # Training data
        train_traj_file = os.path.join(train_data_dir, "traj_w_c.txt")
        self.rgb_train_dir = os.path.join(train_data_dir, "rgb")
        self.depth_train_dir = os.path.join(
            train_data_dir, "depth"
        )  # depth is in mm uint

        if not use_GT_sem:
            self.semantic_class_train_dir = os.path.join(
                train_data_dir, "semantic_class"
            )
        else:
            self.semantic_class_train_dir = os.path.join(
                train_data_dir, "semantic_class_gt"
            )

        self.semantic_instance_train_dir = os.path.join(
            train_data_dir, "semantic_instance"
        )
        if not os.path.exists(self.semantic_instance_train_dir):
            self.semantic_instance_train_dir = None

        self.Ts_train_full = np.loadtxt(train_traj_file, delimiter=" ").reshape(
            -1, 4, 4
        )
        self.rgb_train_list = sorted(
            glob.glob(self.rgb_train_dir + "/rgb*.png"),
            key=lambda file_name: int(file_name.split("_")[-1][:-4]),
        )
        self.depth_train_list = sorted(
            glob.glob(self.depth_train_dir + "/depth*.png"),
            key=lambda file_name: int(file_name.split("_")[-1][:-4]),
        )
        self.semantic_train_list = sorted(
            glob.glob(self.semantic_class_train_dir + "/semantic_class_*.png"),
            key=lambda file_name: int(file_name.split("_")[-1][:-4]),
        )

        if self.semantic_instance_train_dir is not None:
            self.instance_train_list = sorted(
                glob.glob(
                    self.semantic_instance_train_dir + "/semantic_instance_*.png"
                ),
                key=lambda file_name: int(file_name.split("_")[-1][:-4]),
            )

        # Test data
        test_traj_file = os.path.join(test_data_dir, "traj_w_c.txt")
        self.rgb_test_dir = os.path.join(test_data_dir, "rgb")
        self.depth_test_dir = os.path.join(
            test_data_dir, "depth"
        )  # depth is in mm uint
        self.semantic_class_test_dir = os.path.join(test_data_dir, "semantic_class")
        self.semantic_instance_test_dir = os.path.join(
            test_data_dir, "semantic_instance"
        )
        if not os.path.exists(self.semantic_instance_test_dir):
            self.semantic_instance_test_dir = None

        self.Ts_test_full = np.loadtxt(test_traj_file, delimiter=" ").reshape(-1, 4, 4)
        self.rgb_test_list = sorted(
            glob.glob(self.rgb_test_dir + "/rgb*.png"),
            key=lambda file_name: int(file_name.split("_")[-1][:-4]),
        )
        self.depth_test_list = sorted(
            glob.glob(self.depth_test_dir + "/depth*.png"),
            key=lambda file_name: int(file_name.split("_")[-1][:-4]),
        )
        self.semantic_test_list = sorted(
            glob.glob(self.semantic_class_test_dir + "/semantic_class_*.png"),
            key=lambda file_name: int(file_name.split("_")[-1][:-4]),
        )

        if self.semantic_instance_test_dir is not None:
            self.instance_test_list = sorted(
                glob.glob(self.semantic_instance_test_dir + "/semantic_instance_*.png"),
                key=lambda file_name: int(file_name.split("_")[-1][:-4]),
            )

        self.img_h = img_h
        self.img_w = img_w

        self.train_samples = {
            "image": [],
            "depth": [],
            "semantic": [],
            "T_wc": [],
            "instance": [],
        }

        self.test_samples = {
            "image": [],
            "depth": [],
            "semantic": [],
            "T_wc": [],
            "instance": [],
        }

        # Training samples
        self.train_num = len(self.rgb_train_list)
        for idx in range(self.train_num):
            image = (
                cv2.imread(self.rgb_train_list[idx])[:, :, ::-1] / 255.0
            )  # change from BGR uinit 8 to RGB float
            depth = (
                cv2.imread(self.depth_train_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0
            )  # uint16 mm depth, then turn depth from mm to meter
            semantic = cv2.imread(self.semantic_train_list[idx], cv2.IMREAD_UNCHANGED)
            semantic += 1  # to differentiate 'background' from 'void' class

            if self.semantic_instance_train_dir is not None:
                instance = cv2.imread(
                    self.instance_train_list[idx], cv2.IMREAD_UNCHANGED
                )  # uint16

            if (self.img_h is not None and self.img_h != image.shape[0]) or (
                self.img_w is not None and self.img_w != image.shape[1]
            ):
                image = cv2.resize(
                    image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR
                )
                depth = cv2.resize(
                    depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR
                )
                semantic = cv2.resize(
                    semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST
                )
                if self.semantic_instance_train_dir is not None:
                    instance = cv2.resize(
                        instance,
                        (self.img_w, self.img_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

            T_wc = self.Ts_train_full[idx]

            self.train_samples["image"].append(image)
            self.train_samples["depth"].append(depth)
            self.train_samples["semantic"].append(semantic)

            if self.semantic_instance_train_dir is not None:
                self.train_samples["instance"].append(instance)
            self.train_samples["T_wc"].append(T_wc)

        # Test samples
        self.test_num = 100
        min_depth, max_depth = None, None
        for idx in range(self.test_num):
            image = (
                cv2.imread(self.rgb_test_list[idx])[:, :, ::-1] / 255.0
            )  # change from BGR uinit 8 to RGB float
            depth = (
                cv2.imread(self.depth_test_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0
            )  # uint16 mm depth, then turn depth from mm to meter

            min_depth_val = depth.min().item()
            if min_depth is None or min_depth_val < min_depth:
                min_depth = min_depth_val
            max_depth_val = depth.max().item()
            if max_depth is None or max_depth_val > max_depth:
                max_depth = max_depth_val

            semantic = cv2.imread(self.semantic_test_list[idx], cv2.IMREAD_UNCHANGED)
            semantic += 1  # to differentiate 'background' from 'void' class

            if self.semantic_instance_test_dir is not None:
                instance = cv2.imread(
                    self.instance_test_list[idx], cv2.IMREAD_UNCHANGED
                )  # uint16

            if (self.img_h is not None and self.img_h != image.shape[0]) or (
                self.img_w is not None and self.img_w != image.shape[1]
            ):
                image = cv2.resize(
                    image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR
                )
                depth = cv2.resize(
                    depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR
                )
                semantic = cv2.resize(
                    semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST
                )
                if self.semantic_instance_test_dir is not None:
                    instance = cv2.resize(
                        instance,
                        (self.img_w, self.img_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
            T_wc = self.Ts_test_full[idx]

            self.test_samples["image"].append(image)
            self.test_samples["depth"].append(depth)
            self.test_samples["semantic"].append(semantic)

            if self.semantic_instance_test_dir is not None:
                self.test_samples["instance"].append(instance)
            self.test_samples["T_wc"].append(T_wc)

        for (
            key
        ) in (
            self.test_samples.keys()
        ):  # transform list of np array to array with batch dimension
            self.train_samples[key] = np.asarray(self.train_samples[key])
            self.test_samples[key] = np.asarray(self.test_samples[key])

        self.semantic_classes = np.unique(
            np.concatenate(
                (
                    np.unique(self.train_samples["semantic"]),
                    np.unique(self.test_samples["semantic"]),
                )
            )
        ).astype(np.uint8)
        self.semantic_classes = np.concatenate(
            ([0], self.semantic_classes)
        )  # adding 'void' class

        self.num_semantic_class = self.semantic_classes.shape[
            0
        ]  # number of semantic classes, including the void class of 0

        self.colour_map_np = color_palette * 255
        self.colour_map_np = self.colour_map_np.astype(np.uint8)

        self.mask_ids = np.ones(self.train_num)  # init self.mask_ids as full ones
        # 1 means the correspinding label map is used for semantic loss during training, while 0 means no semantic loss is applied on this frame

        # remap existing semantic class labels to continuous label ranging from 0 to num_class-1
        self.train_samples["semantic_clean"] = self.train_samples["semantic"].copy()
        self.train_samples["semantic_remap"] = self.train_samples["semantic"].copy()
        self.train_samples["semantic_remap_clean"] = self.train_samples[
            "semantic_clean"
        ].copy()

        self.test_samples["semantic_remap"] = self.test_samples["semantic"].copy()

        for i in range(self.num_semantic_class):
            self.train_samples["semantic_remap"][
                self.train_samples["semantic"] == self.semantic_classes[i]
            ] = i
            self.train_samples["semantic_remap_clean"][
                self.train_samples["semantic_clean"] == self.semantic_classes[i]
            ] = i
            self.test_samples["semantic_remap"][
                self.test_samples["semantic"] == self.semantic_classes[i]
            ] = i

        print()
        print("Training Sample Summary:")
        for key in self.train_samples.keys():
            print(
                f"{key} has shape of {self.train_samples[key].shape}, type {self.train_samples[key].dtype}."
            )
        print()
        print("Testing Sample Summary:")
        for key in self.test_samples.keys():
            print(
                f"{key} has shape of {self.test_samples[key].shape}, type {self.test_samples[key].dtype}."
            )
