###############################################################################
# Code adapted from https://github.com/devendrachaplot/Object-Goal-Navigation #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                         #
###############################################################################

import json
import bz2
import gzip
import _pickle as cPickle
import gym
import numpy as np
import os
import quaternion
import skimage.morphology
import habitat

from envs.utils.fmm_planner import FMMPlanner
from constants import coco_categories
import envs.utils.pose as pu


class AutoNeRF_Env(habitat.RLEnv):
    """The AutoNeRF environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, args, rank, config_env, dataset):
        self.args = args
        self.rank = rank

        super().__init__(config_env, dataset)

        # Loading dataset info file
        self.split = config_env.DATASET.SPLIT
        self.episodes_dir = config_env.DATASET.EPISODES_DIR.format(split=self.split)

        dataset_info_file = self.episodes_dir + f"{self.split}_info.pbz2"
        with bz2.BZ2File(dataset_info_file, "rb") as f:
            self.dataset_info = cPickle.load(f)

        # Specifying action and observation space
        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(
            0, 255, (3, args.frame_height, args.frame_width), dtype="uint8"
        )

        # Initializations
        self.episode_no = 0

        # Scene info
        self.last_scene_path = None
        self.scene_path = None
        self.scene_name = None

        # Episode Dataset info
        self.eps_data = None
        self.eps_data_idx = None
        self.gt_planner = None
        self.object_boundary = None
        self.map_obj_origin = None
        self.starting_loc = None
        self.starting_distance = None

        # Episode tracking info
        self.curr_distance = None
        self.prev_distance = None
        self.timestep = None
        self.stopped = None
        self.path_length = None
        self.last_sim_location = None
        self.trajectory_states = []
        self.info = {}

        # Saving NeRF training data
        if args.save_autonerf_data == 1:
            self.scene2floors = {}
        else:
            self.scene2floors = None

        self.coco_categories_autonerf = {
            "chair": 0,
            "couch": 1,
            "potted plant": 2,
            "bed": 3,
            "toilet": 4,
            "tv": 5,
            "dining table": 6,
            "oven": 7,
            "sink": 8,
            "refrigerator": 9,
            "book": 10,
            "clock": 11,
            "vase": 12,
            "cup": 13,
            "bottle": 14,
        }

    def load_new_episode(self):
        """The function loads a fixed episode from the episode dataset. This
        function is used for evaluating a trained model on the val split.
        """

        args = self.args
        self.scene_path = self.habitat_env.sim.config.SCENE
        scene_name = self.scene_path.split("/")[-1].split(".")[0]

        if self.scene_path != self.last_scene_path:
            episodes_file = self.episodes_dir + f"content/{scene_name}_episodes.json.gz"

            print(f"Loading episodes from: {episodes_file}")
            with gzip.open(episodes_file, "r") as f:
                self.eps_data = json.loads(f.read().decode("utf-8"))["episodes"]

            self.eps_data_idx = 0
            self.last_scene_path = self.scene_path

        # Load episode info
        episode = self.eps_data[self.eps_data_idx]
        self.eps_data_idx += 1
        self.eps_data_idx = self.eps_data_idx % len(self.eps_data)
        pos = episode["start_position"]
        rot = quaternion.from_float_array(episode["start_rotation"])

        self._env.sim.set_agent_state(pos, rot)
        obs = self._env.sim.get_observations_at(pos, rot)
        return obs

    def sim_map_to_sim_continuous(self, coords):
        """Converts ground-truth 2D Map coordinates to absolute Habitat
        simulator position and rotation.
        """
        agent_state = self._env.sim.get_agent_state(0)
        y, x = coords
        min_x, min_y = self.map_obj_origin / 100.0

        cont_x = x / 20.0 + min_x
        cont_y = y / 20.0 + min_y
        agent_state.position[0] = cont_y
        agent_state.position[2] = cont_x

        rotation = agent_state.rotation
        rvec = quaternion.as_rotation_vector(rotation)

        if self.args.train_single_eps:
            rvec[1] = 0.0
        else:
            rvec[1] = np.random.rand() * 2 * np.pi
        rot = quaternion.from_rotation_vector(rvec)

        return agent_state.position, rot

    def sim_continuous_to_sim_map(self, sim_loc):
        """Converts absolute Habitat simulator pose to ground-truth 2D Map
        coordinates.
        """
        x, y, o = sim_loc
        min_x, min_y = self.map_obj_origin / 100.0
        x, y = int((-x - min_x) * 20.0), int((-y - min_y) * 20.0)

        o = np.rad2deg(o) + 180.0
        return y, x, o

    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, and additional required info
        """
        args = self.args
        new_scene = self.episode_no % args.num_train_episodes == 0

        self.episode_no += 1

        # Initializations
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []

        if new_scene:
            obs = super().reset()
            self.scene_name = self.habitat_env.sim.config.SCENE
            print(f"Changing scene: {self.rank}/{self.scene_name}")

        self.scene_path = self.habitat_env.sim.config.SCENE

        if self.split in ["val", "train"]:
            obs = self.load_new_episode()
        else:
            assert False

        rgb = obs["rgb"].astype(np.uint8)
        depth = obs["depth"]
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.last_sim_location = self.get_sim_location()

        # Set info
        self.info["time"] = self.timestep
        self.info["sensor_pose"] = [0.0, 0.0, 0.0]

        if self.scene2floors is not None:
            self.info["scene_name"] = self.scene_path.split("/")[-1].replace(".glb", "")
            self.info["sim_location"] = self.get_sim_location()
            self.info["floor_level"] = 0

        if "semantic" in obs.keys():
            sem_gt = obs["semantic"]
            instance_sem_gt = sem_gt.copy()

            # Re-mapping sem_gt
            scene = super().habitat_env.sim.semantic_annotations()
            instance_id_to_class_name = {
                int(obj.id.split("_")[-1]): obj.category.name()
                for obj in scene.objects
                if obj is not None
            }
            instance_id_to_coco_id = {}
            for class_id in instance_id_to_class_name.keys():
                class_name = instance_id_to_class_name[class_id]
                if class_name in self.coco_categories_autonerf:
                    instance_id_to_coco_id[class_id] = (
                        self.coco_categories_autonerf[class_name] + 1
                    )
                else:
                    instance_id_to_coco_id[class_id] = 0

            for r in range(sem_gt.shape[0]):
                for c in range(sem_gt.shape[1]):
                    if sem_gt[r, c] != 0:  # if not 'background'
                        sem_gt[r, c] = instance_id_to_coco_id[sem_gt[r, c]]
        else:
            sem_gt = np.zeros((1))
            instance_sem_gt = np.zeros((1))

        return state, self.info, sem_gt, instance_sem_gt

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, and additional required info
        """
        action = action["action"]
        if action == 0:
            self.stopped = True
            # Not sending stop to simulator, resetting manually
            action = 3

        obs, rew, done, _ = super().step(action)

        dx, dy, do = self.get_pose_change()
        self.info["sensor_pose"] = [dx, dy, do]
        self.path_length += pu.get_l2_distance(0, dx, 0, dy)

        rgb = obs["rgb"].astype(np.uint8)
        depth = obs["depth"]
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)

        self.timestep += 1
        self.info["time"] = self.timestep

        if self.scene2floors is not None:
            self.info["scene_name"] = self.scene_path.split("/")[-1].replace(".glb", "")
            self.info["sim_location"] = self.get_sim_location()
            self.info["floor_level"] = 0

        if "semantic" in obs.keys():
            sem_gt = obs["semantic"]
            instance_sem_gt = sem_gt.copy()

            # Re-mapping sem_gt
            scene = super().habitat_env.sim.semantic_annotations()
            instance_id_to_class_name = {
                int(obj.id.split("_")[-1]): obj.category.name()
                for obj in scene.objects
                if obj is not None
            }
            instance_id_to_coco_id = {}
            for class_id in instance_id_to_class_name.keys():
                class_name = instance_id_to_class_name[class_id]
                if class_name in self.coco_categories_autonerf:
                    instance_id_to_coco_id[class_id] = (
                        self.coco_categories_autonerf[class_name] + 1
                    )
                else:
                    instance_id_to_coco_id[class_id] = 0

            for r in range(sem_gt.shape[0]):
                for c in range(sem_gt.shape[1]):
                    if sem_gt[r, c] != 0:  # if not 'background'
                        sem_gt[r, c] = instance_id_to_coco_id[sem_gt[r, c]]
        else:
            sem_gt = np.zeros((1))
            instance_sem_gt = np.zeros((1))

        return state, rew, done, self.info, sem_gt, instance_sem_gt

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0.0, 1.0)

    def get_reward(self, observations):
        return 0.0

    def get_done(self, observations):
        if self.info["time"] >= self.args.max_episode_length - 1:
            done = True
        elif self.stopped:
            done = True
        else:
            done = False
        return done

    def get_info(self, observations):
        """This function is not used, Habitat-RLEnv requires this function"""
        info = {}
        return info

    def get_spaces(self):
        """Returns observation and action spaces."""
        return self.observation_space, self.action_space

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do
