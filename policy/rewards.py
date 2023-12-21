import torch


def compute_reward(
    args,
    device,
    num_scenes,
    full_map_prev,
    full_map,
    use_expl_reward,
    use_sem_obj_reward,
    use_obs_reward,
    use_viewpoint_reward,
):
    reward = torch.zeros(num_scenes).to(device)
    for i in range(reward.shape[0]):
        if use_expl_reward:
            nb_voxels_prev = full_map_prev[i, 1].sum(1).sum(0)
            nb_voxels_new = full_map[i, 1].sum(1).sum(0)
        elif use_sem_obj_reward:
            nb_voxels_prev = full_map_prev[i, 4:20].sum()
            nb_voxels_new = full_map[i, 4:20].sum()
        elif use_obs_reward:
            nb_voxels_prev = full_map_prev[i, 0].sum()
            nb_voxels_new = full_map[i, 0].sum()
        elif use_viewpoint_reward:
            nb_voxels_prev = full_map_prev[i, 20:].sum()
            nb_voxels_new = full_map[i, 20:].sum()
        else:
            return reward

        assert nb_voxels_new >= nb_voxels_prev
        reward[i] = nb_voxels_new - nb_voxels_prev

    if use_expl_reward:
        reward *= (args.map_resolution / 100.0) ** 2  # to m^2
        reward *= args.expl_rew_coeff
    elif use_sem_obj_reward:
        reward *= args.sem_obj_rew_coeff
    elif use_obs_reward:
        reward *= args.obs_rew_coeff
    elif use_viewpoint_reward:
        reward *= args.viewpoint_rew_coeff

    return reward
