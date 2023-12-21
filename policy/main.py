###############################################################################
# Code adapted from https://github.com/devendrachaplot/Object-Goal-Navigation #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                         #
###############################################################################

from collections import deque
import gym
import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn

import algo
from arguments import get_args
from envs import make_vec_envs
from model import RL_Policy, Semantic_Mapping
from nerf_data_saving import saving_autonerf_data
from rewards import compute_reward
from utils.storage import GlobalRolloutStorage

os.environ["OMP_NUM_THREADS"] = "1"


def main():
    args = get_args()

    # Expl. reward
    use_expl_reward = args.use_expl_reward == 1

    # Obs. reward
    use_obs_reward = args.use_obs_reward == 1

    # Sem. obj. reward
    use_sem_obj_reward = args.use_sem_obj_reward == 1

    # View. reward
    use_viewpoint_reward = args.use_viewpoint_reward == 1
    if use_viewpoint_reward:
        assert args.nb_additional_channels in [4, 8, 12]
        nb_additional_channels = args.nb_additional_channels
    else:
        nb_additional_channels = 0

    if args.eval:
        reset_l_step = False
        l_step = 0

    # Saving NeRF training data
    save_autonerf_data = args.save_autonerf_data == 1
    if save_autonerf_data:
        # Only saving data on val scenes
        assert args.eval == 1
        # Path to the dataset where saving rgb, depth and semantics
        autonerf_dataset_path = args.autonerf_dataset_path

        # Scene to traj id dict
        scene2traj_id = {
            "Collierville": 0,
            "Corozal": 0,
            "Darden": 0,
            "Markleeville": 0,
            "Wiconisco": 0,
        }

        # Sets of poses (to remove duplicates)
        pose_sets = [[] for _ in range(args.num_processes)]

        # Floor levels at the beginning of the episode
        floor_levels = [-1 for _ in range(args.num_processes)]

        # Frame ids
        frame_ids = np.zeros((args.num_processes))

        # Max number of steps
        max_num_steps_autonerf = 10 * args.max_episode_length

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup Logging
    log_dir = f"{args.dump_location}/models/{args.exp_name}/"
    dump_dir = f"{args.dump_location}/dump/{args.exp_name}/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    logging.basicConfig(filename=log_dir + "train.log", level=logging.INFO)
    print(f"Dumping at {log_dir}")
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    g_masks = torch.ones(num_scenes).float().to(device)
    best_g_reward = -np.inf

    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    g_episode_rewards = deque(maxlen=1000)
    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)
    per_step_g_rewards = deque(maxlen=1000)
    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos, autonerf_data, sem_gt, instance_sem_gt = envs.reset()

    if save_autonerf_data:
        for env_idx in range(args.num_processes):
            floor_levels[env_idx] = infos[env_idx]["floor_level"]
        frame_ids = saving_autonerf_data(
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
        )

    torch.set_grad_enabled(False)

    # Initialize map variables:
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    # nc = args.num_sem_categories + 4  # num channels

    nc = (
        args.num_sem_categories + 4 + nb_additional_channels
    )  # num_sem_classes + 4 (obstacle, explored, current loc, past loc)
    # + 'nb_additional_channels' (each cell contains 'nb_additional_channels' channels for each of the 12 bins
    # corresponding to possible viewpoint directions)

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w, local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose():
        full_map.fill_(0.0)
        full_pose.fill_(0.0)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / args.map_resolution),
                int(c * 100.0 / args.map_resolution),
            ]

            full_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries(
                (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
            )

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [
                lmb[e][2] * args.map_resolution / 100.0,
                lmb[e][0] * args.map_resolution / 100.0,
                0.0,
            ]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]]
            local_pose[e] = (
                full_pose[e] - torch.from_numpy(origins[e]).to(device).float()
            )

    def init_map_and_pose_for_env(e):
        full_map[e].fill_(0.0)
        full_pose[e].fill_(0.0)
        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [
            int(r * 100.0 / args.map_resolution),
            int(c * 100.0 / args.map_resolution),
        ]

        full_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries(
            (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
        )

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [
            lmb[e][2] * args.map_resolution / 100.0,
            lmb[e][0] * args.map_resolution / 100.0,
            0.0,
        ]

        local_map[e] = full_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]]
        local_pose[e] = full_pose[e] - torch.from_numpy(origins[e]).to(device).float()

    init_map_and_pose()

    # Global policy observation space
    ngc = 8 + args.num_sem_categories + 2 * nb_additional_channels
    es = 1

    g_observation_space = gym.spaces.Box(0, 1, (ngc, local_w, local_h), dtype="uint8")

    # Global policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=0.99, shape=(2,), dtype=np.float32)

    # Global policy recurrent layer size
    g_hidden_size = args.global_hidden_size

    # Semantic Mapping
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()

    # Global policy
    g_policy = RL_Policy(
        g_observation_space.shape,
        g_action_space,
        model_type=1,
        base_kwargs={
            "recurrent": args.use_recurrent_global,
            "hidden_size": g_hidden_size,
            "num_sem_categories": ngc - 8,
        },
    ).to(device)

    if not args.eval:
        g_agent = algo.PPO(
            g_policy,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
        )

    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()
    extras = torch.zeros(num_scenes, 1)

    # Storage
    g_rollouts = GlobalRolloutStorage(
        args.num_global_steps,
        num_scenes,
        g_observation_space.shape,
        g_action_space,
        g_policy.rec_state_size,
        es,
    ).to(device)

    if args.load != "0":
        print(f"Loading model {args.load}")
        state_dict = torch.load(args.load, map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)

    if args.eval:
        g_policy.eval()

    # Predict semantic map from frame 1
    poses = (
        torch.from_numpy(
            np.asarray([infos[env_idx]["sensor_pose"] for env_idx in range(num_scenes)])
        )
        .float()
        .to(device)
    )

    _, local_map, _, local_pose = sem_map_module(obs, poses, local_map, local_pose)

    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [
            int(r * 100.0 / args.map_resolution),
            int(c * 100.0 / args.map_resolution),
        ]

        local_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.0)

    global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()
    global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
        full_map[:, 0:4, :, :]
    )

    if nb_additional_channels > 0:
        global_input[:, 8 : 8 + nb_additional_channels, :, :] = local_map[
            :, 20:, :, :
        ].detach()
        global_input[
            :, 8 + nb_additional_channels : 8 + 2 * nb_additional_channels, :, :
        ] = nn.MaxPool2d(args.global_downscaling)(full_map[:, 20:, :, :])
    global_input[:, 8 + 2 * nb_additional_channels :, :, :] = local_map[
        :, 4:20, :, :
    ].detach()

    extras = torch.zeros(num_scenes, 1)
    extras[:, 0] = global_orientation[:, 0]
    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.extras[0].copy_(extras)

    # Run Global Policy (global_goals = Long-Term Goal)
    g_value, g_action, g_action_log_prob, g_rec_states = g_policy.act(
        g_rollouts.obs[0],
        g_rollouts.rec_states[0],
        g_rollouts.masks[0],
        extras=g_rollouts.extras[0],
        deterministic=False,
    )

    cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
    global_goals = [
        [int(action[0] * local_w), int(action[1] * local_h)] for action in cpu_actions
    ]
    global_goals = [
        [min(x, int(local_w - 1)), min(y, int(local_h - 1))] for x, y in global_goals
    ]

    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

    for e in range(num_scenes):
        goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1

    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input["map_pred"] = local_map[e, 0, :, :].cpu().numpy()
        p_input["exp_pred"] = local_map[e, 1, :, :].cpu().numpy()
        p_input["pose_pred"] = planner_pose_inputs[e]
        p_input["goal"] = goal_maps[e]
        p_input["new_goal"] = 1
        p_input["wait"] = wait_env[e] or finished[e]
        if args.visualize or args.print_images:
            p_input["sem_map_pred"] = local_map[e, 4:20, :, :].argmax(0).cpu().numpy()

            if use_viewpoint_reward:
                p_input["sem_map_viewpoints"] = (
                    local_map[e, 20:, :, :].cpu().numpy(),
                    origins,
                )

    (
        obs,
        _,
        done,
        infos,
        autonerf_data,
        sem_gt,
        instance_sem_gt,
    ) = envs.plan_act_and_preprocess(planner_inputs)

    start = time.time()
    torch.set_grad_enabled(False)
    for step in range(args.num_training_frames // args.num_processes + 1):
        if finished.sum() == args.num_processes:
            break

        g_step = (step // args.num_local_steps) % args.num_global_steps

        if not args.eval:
            l_step = step % args.num_local_steps
        else:
            assert args.num_processes == 1
            if done[0]:
                reset_l_step = True

        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        l_masks = torch.FloatTensor([0 if x else 1 for x in done]).to(device)
        g_masks *= l_masks

        # Resetting ids when done
        if save_autonerf_data:
            for env_idx in range(num_scenes):
                if done[env_idx]:
                    scene_name = infos[env_idx]["scene_name"]
                    scene2traj_id[scene_name] += 1
                    pose_sets[env_idx] = []
                    frame_ids[env_idx] = 0
                    floor_levels[env_idx] = infos[env_idx]["floor_level"]

            frame_ids = saving_autonerf_data(
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
            )

        for e, x in enumerate(done):
            if x:
                wait_env[e] = 1.0
                init_map_and_pose_for_env(e)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Semantic Mapping Module
        poses = (
            torch.from_numpy(
                np.asarray(
                    [infos[env_idx]["sensor_pose"] for env_idx in range(num_scenes)]
                )
            )
            .float()
            .to(device)
        )

        _, local_map, _, local_pose = sem_map_module(obs, poses, local_map, local_pose)

        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.0)  # Resetting current location channel

        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / args.map_resolution),
                int(c * 100.0 / args.map_resolution),
            ]
            local_map[e, 2:4, loc_r - 2 : loc_r + 3, loc_c - 2 : loc_c + 3] = 1.0

        # ------------------------------------------------------------------

        # At eval time, if agent close enough to goal, re-sample a new goal (i.e. set l_step to args.num_local_steps - 1)
        if args.eval:
            assert args.num_processes == 1

            if reset_l_step:
                l_step = 0
                reset_l_step = False
            else:
                l_step += 1

            locs_global_goals = np.zeros((locs[:, :2].shape))
            locs_global_goals[0, 0] = (global_goals[0][1] * args.map_resolution) / 100.0
            locs_global_goals[0, 1] = (global_goals[0][0] * args.map_resolution) / 100.0

            dist_to_goal = np.linalg.norm(locs[:, :2] - locs_global_goals)

            if dist_to_goal <= 1.0:
                l_step = args.num_local_steps - 1

        # ------------------------------------------------------------------
        # Global Policy
        if l_step == args.num_local_steps - 1:
            if args.eval:
                reset_l_step = True

            if (
                use_expl_reward
                or use_obs_reward
                or use_sem_obj_reward
                or use_viewpoint_reward
                or save_autonerf_data
            ):
                full_map_prev = full_map.clone()

            # For every global step, update the full and local maps
            for e in range(num_scenes):
                if wait_env[e] == 1:  # New episode
                    wait_env[e] = 0.0
                full_map[
                    e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
                ] = local_map[e]
                full_pose[e] = (
                    local_pose[e] + torch.from_numpy(origins[e]).to(device).float()
                )

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [
                    int(r * 100.0 / args.map_resolution),
                    int(c * 100.0 / args.map_resolution),
                ]

                lmb[e] = get_local_map_boundaries(
                    (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
                )

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [
                    lmb[e][2] * args.map_resolution / 100.0,
                    lmb[e][0] * args.map_resolution / 100.0,
                    0.0,
                ]

                local_map[e] = full_map[
                    e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]
                ]
                local_pose[e] = (
                    full_pose[e] - torch.from_numpy(origins[e]).to(device).float()
                )

            locs = local_pose.cpu().numpy()
            for e in range(num_scenes):
                global_orientation[e] = int((locs[e, 2] + 180.0) / 5.0)
            global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :]
            global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
                full_map[:, 0:4, :, :]
            )

            if nb_additional_channels > 0:
                global_input[:, 8 : 8 + nb_additional_channels, :, :] = local_map[
                    :, 20:, :, :
                ].detach()
                global_input[
                    :, 8 + nb_additional_channels : 8 + 2 * nb_additional_channels, :, :
                ] = nn.MaxPool2d(args.global_downscaling)(full_map[:, 20:, :, :])
            global_input[:, 8 + 2 * nb_additional_channels :, :, :] = local_map[
                :, 4:20, :, :
            ].detach()
            extras[:, 0] = global_orientation[:, 0]

            # Computing reward
            g_reward = compute_reward(
                args,
                device,
                num_scenes,
                full_map_prev,
                full_map,
                use_expl_reward,
                use_sem_obj_reward,
                use_obs_reward,
                use_viewpoint_reward,
            )

            g_process_rewards += g_reward.cpu().numpy()
            g_total_rewards = g_process_rewards * (1 - g_masks.cpu().numpy())
            g_process_rewards *= g_masks.cpu().numpy()
            per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

            if np.sum(g_total_rewards) != 0:
                for total_rew in g_total_rewards:
                    if total_rew != 0:
                        g_episode_rewards.append(total_rew)

            # Add samples to global policy storage
            if step == 0:
                g_rollouts.obs[0].copy_(global_input)
                g_rollouts.extras[0].copy_(extras)
            else:
                g_rollouts.insert(
                    global_input,
                    g_rec_states,
                    g_action,
                    g_action_log_prob,
                    g_value,
                    g_reward,
                    g_masks,
                    extras,
                )

            # Sample long-term goal from global policy
            g_value, g_action, g_action_log_prob, g_rec_states = g_policy.act(
                g_rollouts.obs[g_step + 1],
                g_rollouts.rec_states[g_step + 1],
                g_rollouts.masks[g_step + 1],
                extras=g_rollouts.extras[g_step + 1],
                deterministic=False,
            )
            cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
            global_goals = [
                [int(action[0] * local_w), int(action[1] * local_h)]
                for action in cpu_actions
            ]
            global_goals = [
                [min(x, int(local_w - 1)), min(y, int(local_h - 1))]
                for x, y in global_goals
            ]
            g_masks = torch.ones(num_scenes).float().to(device)

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

        for e in range(num_scenes):
            goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input["map_pred"] = local_map[e, 0, :, :].cpu().numpy()
            p_input["exp_pred"] = local_map[e, 1, :, :].cpu().numpy()
            p_input["pose_pred"] = planner_pose_inputs[e]
            p_input["goal"] = goal_maps[e]
            p_input["new_goal"] = l_step == args.num_local_steps - 1
            p_input["wait"] = wait_env[e] or finished[e]
            if args.visualize or args.print_images:
                p_input["sem_map_pred"] = (
                    local_map[e, 4:20, :, :].argmax(0).cpu().numpy()
                )

                if use_viewpoint_reward:
                    p_input["sem_map_viewpoints"] = (
                        local_map[e, 20:, :, :].cpu().numpy(),
                        origins,
                    )

        (
            obs,
            _,
            done,
            infos,
            autonerf_data,
            sem_gt,
            instance_sem_gt,
        ) = envs.plan_act_and_preprocess(planner_inputs)

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Training
        torch.set_grad_enabled(True)
        if (
            g_step % args.num_global_steps == args.num_global_steps - 1
            and l_step == args.num_local_steps - 1
        ):
            if not args.eval:
                g_next_value = g_policy.get_value(
                    g_rollouts.obs[-1],
                    g_rollouts.rec_states[-1],
                    g_rollouts.masks[-1],
                    extras=g_rollouts.extras[-1],
                ).detach()

                g_rollouts.compute_returns(
                    g_next_value, args.use_gae, args.gamma, args.tau
                )
                g_value_loss, g_action_loss, g_dist_entropy = g_agent.update(g_rollouts)
                g_value_losses.append(g_value_loss)
                g_action_losses.append(g_action_loss)
                g_dist_entropies.append(g_dist_entropy)
            g_rollouts.after_update()

        torch.set_grad_enabled(False)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Logging
        if step % args.log_interval == 0:
            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join(
                [
                    "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                    "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                    "num timesteps {},".format(step * num_scenes),
                    "FPS {},".format(int(step * num_scenes / (end - start))),
                ]
            )

            log += "\n\tRewards:"
            if len(g_episode_rewards) > 0:
                log += " ".join(
                    [
                        " Global step mean/med rew:",
                        "{:.4f}/{:.4f},".format(
                            np.mean(per_step_g_rewards), np.median(per_step_g_rewards)
                        ),
                        " Global eps mean/med/min/max eps rew:",
                        "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_episode_rewards),
                            np.median(g_episode_rewards),
                            np.min(g_episode_rewards),
                            np.max(g_episode_rewards),
                        ),
                    ]
                )
            log += "\n\tLosses:"
            if len(g_value_losses) > 0 and not args.eval:
                log += " ".join(
                    [
                        " Policy Loss value/action/dist:",
                        "{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_value_losses),
                            np.mean(g_action_losses),
                            np.mean(g_dist_entropies),
                        ),
                    ]
                )

            print(log)
            logging.info(log)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Save best models
        if (step * num_scenes) % args.save_interval < num_scenes:
            if (
                len(g_episode_rewards) >= 1000
                and (np.mean(g_episode_rewards) >= best_g_reward)
                and not args.eval
            ):
                torch.save(
                    g_policy.state_dict(), os.path.join(log_dir, "model_best.pth")
                )
                best_g_reward = np.mean(g_episode_rewards)

        # Save periodic models
        if (step * num_scenes) % args.save_periodic < num_scenes:
            total_steps = step * num_scenes
            if not args.eval:
                torch.save(
                    g_policy.state_dict(),
                    os.path.join(dump_dir, f"periodic_{total_steps}.pth"),
                )
        # ------------------------------------------------------------------

        # Stopping when reaching max number of steps
        if save_autonerf_data and step == max_num_steps_autonerf:
            break


if __name__ == "__main__":
    main()
