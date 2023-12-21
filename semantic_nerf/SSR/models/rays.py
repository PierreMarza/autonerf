################################################################
# Code adapted from https://github.com/Harry-Zhi/semantic_nerf #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)          #
################################################################

import numpy as np
import torch

# Ray helpers
def get_rays_camera(B, H, W, fx, fy, cx, cy, depth_type, convention="opencv"):

    assert depth_type is "z" or depth_type is "euclidean"
    i, j = torch.meshgrid(
        torch.arange(W), torch.arange(H)
    )  # pytorch's meshgrid has indexing='ij', we transpose to "xy" moode

    i = i.t().float()
    j = j.t().float()

    size = [B, H, W]

    i_batch = torch.empty(size)
    j_batch = torch.empty(size)
    i_batch[:, :, :] = i[None, :, :]
    j_batch[:, :, :] = j[None, :, :]

    if convention == "opencv":
        x = (i_batch - cx) / fx
        y = (j_batch - cy) / fy
        z = torch.ones(size)
    elif convention == "opengl":
        x = (i_batch - cx) / fx
        y = -(j_batch - cy) / fy
        z = -torch.ones(size)
    else:
        assert False

    dirs = torch.stack((x, y, z), dim=3)  # shape of [B, H, W, 3]

    if depth_type == "euclidean":
        norm = torch.norm(dirs, dim=3, keepdim=True)
        dirs = dirs * (1.0 / norm)

    return dirs


def get_rays_world(T_WC, dirs_C):
    R_WC = T_WC[:, :3, :3]  # Bx3x3
    dirs_W = torch.matmul(R_WC[:, None, ...], dirs_C[..., None]).squeeze(-1)
    origins = T_WC[:, :3, -1]  # Bx3
    origins = torch.broadcast_tensors(origins[:, None, :], dirs_W)[0]
    return origins, dirs_W


def sampling_index(n_rays, batch_size, h, w):

    index_b = np.random.choice(np.arange(batch_size)).reshape(
        (1, 1)
    )  # sample one image from the full trainiing set
    index_hw = torch.randint(0, h * w, (1, n_rays))

    return index_b, index_hw


# Hierarchical sampling using inverse CDF transformations
def sample_pdf(bins, weights, N_samples, det=False):
    """Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: N_rays x (N_samples_coarse - 1)
        weights: N_rays x (N_samples_coarse - 2)
        N_samples: N_samples_fine
        det: deterministic or not
    """
    # Get pdf
    weights = (
        weights + 1e-5
    )  # prevent nans, prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)  # N_rays x (N_samples - 2)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf], -1
    )  # N_rays x (N_samples_coarse - 1)
    # padded to 0~1 inclusive, (N_rays, N_samples-1)

    # Take uniform samples
    if det:  # generate deterministic samples
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)
        # (N_rays, N_samples_fine)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)  # N_rays x N_samples_fine
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (N_rays, N_samples_fine, 2)

    matched_shape = [
        inds_g.shape[0],
        inds_g.shape[1],
        cdf.shape[-1],
    ]  # (N_rays, N_samples_fine, N_samples_coarse - 1)

    cdf_g = torch.gather(
        cdf.unsqueeze(1).expand(matched_shape), 2, inds_g
    )  # N_rays, N_samples_fine, 2
    bins_g = torch.gather(
        bins.unsqueeze(1).expand(matched_shape), 2, inds_g
    )  # N_rays, N_samples_fine, 2

    denom = cdf_g[..., 1] - cdf_g[..., 0]  # # N_rays, N_samples_fine
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def create_rays(
    num_rays,
    Ts_c2w,
    height,
    width,
    fx,
    fy,
    cx,
    cy,
    near,
    far,
    c2w_staticcam=None,
    depth_type="z",
    use_viewdirs=True,
    convention="opencv",
):
    """
    convention:
    "opencv" or "opengl". It defines the coordinates convention of rays from cameras.
    OpenCv defines x,y,z as right, down, forward while OpenGl defines x,y,z as right, up, backward (camera looking towards forward direction still, -z!)
    Note: Use either convention is fine, but the corresponding pose should follow the same convention.

    """
    rays_cam = get_rays_camera(
        num_rays,
        height,
        width,
        fx,
        fy,
        cx,
        cy,
        depth_type=depth_type,
        convention=convention,
    )  # [N, H, W, 3]

    dirs_C = rays_cam.view(num_rays, -1, 3)  # [N, HW, 3]
    rays_o, rays_d = get_rays_world(
        Ts_c2w, dirs_C
    )  # origins: [B, HW, 3], dirs_W: [B, HW, 3]

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # c2w_staticcam: If not None, use this transformation matrix for camera,
            # while using other c2w argument for viewing directions.
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays_world(
                c2w_staticcam, dirs_C
            )  # origins: [B, HW, 3], dirs_W: [B, HW, 3]

        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
        rays_d[..., :1]
    )
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    return rays
