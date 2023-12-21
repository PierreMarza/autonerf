################################################################
# Code adapted from https://github.com/Harry-Zhi/semantic_nerf #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)          #
################################################################

from collections import defaultdict
import copy
import imageio
from imgviz import depth2rgb
import json
import logging
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml

from SSR.models.model_utils import raw2outputs
from SSR.models.model_utils import run_network
from SSR.models.rays import sampling_index, sample_pdf, create_rays
from SSR.models.semantic_nerf import get_embedder, Semantic_NeRF
from SSR.training.training_utils import (
    batchify_rays,
    calculate_segmentation_metrics,
    calculate_depth_metrics,
)
from SSR.utils import image_utils
from SSR.visualisation.tensorboard_vis import TFVisualizer


def select_gpus(gpus):
    """
    takes in a string containing a comma-separated list
    of gpus to make visible to tensorflow, e.g. '0,1,3'
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpus is not "":
        logging.info(f"Using gpu's: {gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    else:
        logging.info("Using all available gpus")


class SSRTrainer(object):
    def __init__(self, config):
        super(SSRTrainer, self).__init__()
        self.config = config
        self.set_params()

        self.training = True  # training mode by default
        # create tfb Summary writers and folders
        tf_log_dir = os.path.join(config["experiment"]["save_dir"], "tfb_logs")
        if not os.path.exists(tf_log_dir):
            os.makedirs(tf_log_dir)
        self.tfb_viz = TFVisualizer(
            tf_log_dir, config["logging"]["step_log_tfb"], config
        )

    def save_config(self):
        # save config to save_dir for the convience of checking config later
        with open(
            os.path.join(self.config["experiment"]["save_dir"], "exp_config.yaml"), "w"
        ) as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

    def set_params_autonerf(self):
        self.H = self.config["experiment"]["height"]
        self.W = self.config["experiment"]["width"]

        self.n_pix = self.H * self.W
        self.aspect_ratio = self.W / self.H

        self.hfov = 90
        # the pin-hole camera has the same value for fx and fy
        self.fx = self.W / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        self.fy = self.fx
        self.cx = (self.W - 1.0) / 2.0
        self.cy = (self.H - 1.0) / 2.0

        self.near, self.far = self.config["render"]["depth_range"]
        self.c2w_staticcam = None

        # use scaled size for test and visualisation purpose
        self.test_viz_factor = int(self.config["render"]["test_viz_factor"])
        self.H_scaled = self.H // self.test_viz_factor
        self.W_scaled = self.W // self.test_viz_factor
        self.fx_scaled = self.W_scaled / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        self.fy_scaled = self.fx_scaled
        self.cx_scaled = (self.W_scaled - 1.0) / 2.0
        self.cy_scaled = (self.H_scaled - 1.0) / 2.0

        self.save_config()

    def set_params(self):
        self.enable_semantic = self.config["experiment"]["enable_semantic"]

        # render options
        self.n_rays = (
            eval(self.config["render"]["N_rays"])
            if isinstance(self.config["render"]["N_rays"], str)
            else self.config["render"]["N_rays"]
        )

        self.N_samples = self.config["render"]["N_samples"]
        self.netchunk = (
            eval(self.config["model"]["netchunk"])
            if isinstance(self.config["model"]["netchunk"], str)
            else self.config["model"]["netchunk"]
        )

        self.chunk = (
            eval(self.config["model"]["chunk"])
            if isinstance(self.config["model"]["chunk"], str)
            else self.config["model"]["chunk"]
        )

        self.use_viewdir = self.config["render"]["use_viewdirs"]

        self.convention = self.config["experiment"]["convention"]

        self.endpoint_feat = (
            self.config["experiment"]["endpoint_feat"]
            if "endpoint_feat" in self.config["experiment"].keys()
            else False
        )

        self.N_importance = self.config["render"]["N_importance"]
        self.raw_noise_std = self.config["render"]["raw_noise_std"]
        self.white_bkgd = self.config["render"]["white_bkgd"]
        self.perturb = self.config["render"]["perturb"]

        self.no_batching = self.config["render"]["no_batching"]

        self.lrate = float(self.config["train"]["lrate"])
        self.lrate_decay = float(self.config["train"]["lrate_decay"])

        # logging
        self.save_dir = self.config["experiment"]["save_dir"]

    def prepare_data_autonerf(self, data, gpu=True):
        self.ignore_label = -1

        # shift numpy data to torch
        train_samples = data.train_samples
        test_samples = data.test_samples
        self.mask_ids = data.mask_ids
        self.num_train = data.train_num
        self.num_test = data.test_num

        # preprocess semantic info
        self.semantic_classes = torch.from_numpy(data.semantic_classes)
        self.num_semantic_class = self.semantic_classes.shape[
            0
        ]  # number of semantic classes, including void class=0
        self.num_valid_semantic_class = (
            self.num_semantic_class - 1
        )  # exclude void class
        assert self.num_semantic_class == data.num_semantic_class

        json_class_mapping = os.path.join(
            self.config["experiment"]["scene_file"], "info_semantic.json"
        )
        with open(json_class_mapping, "r") as f:
            annotations = json.load(f)

        colour_map_np = data.colour_map_np
        self.colour_map = torch.from_numpy(colour_map_np)
        self.valid_colour_map = torch.from_numpy(
            colour_map_np[1:, :]
        )  # exclude the first colour map to colourise rendered segmentation without void index

        class_name_string = ["void"] + [x["name"] for x in annotations["classes"]]
        legend_img_arr = image_utils.plot_semantic_legend(
            data.semantic_classes,
            class_name_string,
            colormap=colour_map_np,
            save_path=self.save_dir,
        )

        # remap different semantic classes to continuous integers from 0 to num_class-1
        self.semantic_classes_remap = torch.from_numpy(
            np.arange(self.num_semantic_class)
        )

        # Training data
        # rgb
        self.train_image = torch.from_numpy(train_samples["image"])
        self.train_image_scaled = F.interpolate(
            self.train_image.permute(
                0,
                3,
                1,
                2,
            ),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="bilinear",
        ).permute(0, 2, 3, 1)
        # depth
        self.train_depth = torch.from_numpy(train_samples["depth"])
        self.viz_train_depth = np.stack(
            [
                depth2rgb(dep, min_value=self.near, max_value=self.far)
                for dep in train_samples["depth"]
            ],
            axis=0,
        )  # [num_test, H, W, 3]
        # process the depth for evaluation purpose
        self.train_depth_scaled = (
            F.interpolate(
                torch.unsqueeze(self.train_depth, dim=1).float(),
                scale_factor=1 / self.config["render"]["test_viz_factor"],
                mode="bilinear",
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )

        # semantic
        self.train_semantic = torch.from_numpy(train_samples["semantic_remap"])
        self.viz_train_semantic = np.stack(
            [colour_map_np[sem] for sem in self.train_semantic], axis=0
        )  # [num_test, H, W, 3]

        self.train_semantic_clean = torch.from_numpy(
            train_samples["semantic_remap_clean"]
        )
        self.viz_train_semantic_clean = np.stack(
            [colour_map_np[sem] for sem in self.train_semantic_clean], axis=0
        )  # [num_test, H, W, 3]

        # process the clean label for evaluation purpose
        self.train_semantic_clean_scaled = F.interpolate(
            torch.unsqueeze(self.train_semantic_clean, dim=1).float(),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="nearest",
        ).squeeze(1)
        self.train_semantic_clean_scaled = (
            self.train_semantic_clean_scaled.cpu().numpy() - 1
        )
        # pose
        self.train_Ts = torch.from_numpy(train_samples["T_wc"]).float()

        # Test data
        # rgb
        self.test_image = torch.from_numpy(test_samples["image"])  # [num_test, H, W, 3]
        # scale the test image for evaluation purpose
        self.test_image_scaled = F.interpolate(
            self.test_image.permute(
                0,
                3,
                1,
                2,
            ),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="bilinear",
        ).permute(0, 2, 3, 1)

        # depth
        self.test_depth = torch.from_numpy(test_samples["depth"])  # [num_test, H, W]
        self.viz_test_depth = np.stack(
            [
                depth2rgb(dep, min_value=self.near, max_value=self.far)
                for dep in test_samples["depth"]
            ],
            axis=0,
        )  # [num_test, H, W, 3]
        # process the depth for evaluation purpose
        self.test_depth_scaled = (
            F.interpolate(
                torch.unsqueeze(self.test_depth, dim=1).float(),
                scale_factor=1 / self.config["render"]["test_viz_factor"],
                mode="bilinear",
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )
        # semantic
        self.test_semantic = torch.from_numpy(
            test_samples["semantic_remap"]
        )  # [num_test, H, W]

        self.viz_test_semantic = np.stack(
            [colour_map_np[sem] for sem in self.test_semantic], axis=0
        )  # [num_test, H, W, 3]

        self.test_semantic_scaled = F.interpolate(
            torch.unsqueeze(self.test_semantic, dim=1).float(),
            scale_factor=1 / self.config["render"]["test_viz_factor"],
            mode="nearest",
        ).squeeze(1)
        self.test_semantic_scaled = (
            self.test_semantic_scaled.cpu().numpy() - 1
        )  # shift void class from value 0 to -1, to match self.ignore_label
        # pose
        self.test_Ts = torch.from_numpy(
            test_samples["T_wc"]
        ).float()  # [num_test, 4, 4]

        # set the data sampling paras which need the number of training images
        if (
            self.no_batching is False
        ):  # False means we need to sample from all rays instead of rays from one random image
            self.i_batch = 0
            self.rand_idx = torch.randperm(self.num_train * self.H * self.W)

        # add datasets to tfboard for comparison to rendered images
        self.tfb_viz.tb_writer.add_image(
            "Train/legend",
            np.expand_dims(legend_img_arr, axis=0),
            0,
            dataformats="NHWC",
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/rgb_GT", train_samples["image"], 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/depth_GT", self.viz_train_depth, 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/vis_sem_label_GT", self.viz_train_semantic, 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Train/vis_sem_label_GT_clean",
            self.viz_train_semantic_clean,
            0,
            dataformats="NHWC",
        )

        self.tfb_viz.tb_writer.add_image(
            "Test/legend", np.expand_dims(legend_img_arr, axis=0), 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Test/rgb_GT", test_samples["image"], 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Test/depth_GT", self.viz_test_depth, 0, dataformats="NHWC"
        )
        self.tfb_viz.tb_writer.add_image(
            "Test/vis_sem_label_GT", self.viz_test_semantic, 0, dataformats="NHWC"
        )

    def init_rays(self):
        # create rays
        rays = create_rays(
            self.num_train,
            self.train_Ts,
            self.H,
            self.W,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.near,
            self.far,
            use_viewdirs=self.use_viewdir,
            convention=self.convention,
        )

        rays_vis = create_rays(
            self.num_train,
            self.train_Ts,
            self.H_scaled,
            self.W_scaled,
            self.fx_scaled,
            self.fy_scaled,
            self.cx_scaled,
            self.cy_scaled,
            self.near,
            self.far,
            use_viewdirs=self.use_viewdir,
            convention=self.convention,
        )

        rays_test = create_rays(
            self.num_test,
            self.test_Ts,
            self.H_scaled,
            self.W_scaled,
            self.fx_scaled,
            self.fy_scaled,
            self.cx_scaled,
            self.cy_scaled,
            self.near,
            self.far,
            use_viewdirs=self.use_viewdir,
            convention=self.convention,
        )

        self.rays = rays  # [num_images, H*W, 11]
        self.rays_vis = rays_vis
        self.rays_test = rays_test

    def sample_data(self, step, rays, h, w, no_batching=True, mode="train"):
        # generate sampling index
        num_img, num_ray, ray_dim = rays.shape

        assert num_ray == h * w
        total_ray_num = num_img * h * w

        if mode == "train":
            image = self.train_image
            if self.enable_semantic:
                depth = self.train_depth
                semantic = self.train_semantic
            sample_num = self.num_train
        elif mode == "test":
            image = self.test_image
            if self.enable_semantic:
                depth = self.test_depth
                semantic = self.test_semantic
            sample_num = self.num_test
        elif mode == "vis":
            assert False
        else:
            assert False

        # sample rays and ground truth data
        sematic_available_flag = 1

        if no_batching:  # sample random pixels from one random images
            index_batch, index_hw = sampling_index(self.n_rays, num_img, h, w)
            sampled_rays = rays[index_batch, index_hw, :]

            flat_sampled_rays = sampled_rays.reshape([-1, ray_dim]).float()
            gt_image = image.reshape(sample_num, -1, 3)[
                index_batch, index_hw, :
            ].reshape(-1, 3)
            if self.enable_semantic:
                gt_depth = depth.reshape(sample_num, -1)[index_batch, index_hw].reshape(
                    -1
                )
                sematic_available_flag = self.mask_ids[
                    index_batch
                ]  # semantic available if mask_id is 1 (train with rgb loss and semantic loss) else 0 (train with rgb loss only)
                gt_semantic = semantic.reshape(sample_num, -1)[
                    index_batch, index_hw
                ].reshape(-1)
                gt_semantic = gt_semantic.cuda()
        else:  # sample from all random pixels
            index_hw = self.rand_idx[self.i_batch : self.i_batch + self.n_rays]

            flat_rays = rays.reshape([-1, ray_dim]).float()
            flat_sampled_rays = flat_rays[index_hw, :]
            gt_image = image.reshape(-1, 3)[index_hw, :]
            if self.enable_semantic:
                gt_depth = depth.reshape(-1)[index_hw]
                gt_semantic = semantic.reshape(-1)[index_hw]
                gt_semantic = gt_semantic.cuda()

            self.i_batch += self.n_rays
            if self.i_batch >= total_ray_num:
                print("Shuffle data after an epoch!")
                self.rand_idx = torch.randperm(total_ray_num)
                self.i_batch = 0

        sampled_rays = flat_sampled_rays
        sampled_gt_rgb = gt_image

        if self.enable_semantic:
            sampled_gt_depth = gt_depth
            sampled_gt_semantic = (
                gt_semantic.long()
            )  # required long type for nn.NLL or nn.crossentropy
            return (
                sampled_rays,
                sampled_gt_rgb,
                sampled_gt_depth,
                sampled_gt_semantic,
                sematic_available_flag,
            )
        else:
            return sampled_rays, sampled_gt_rgb

    def render_rays(self, flat_rays):
        """
        Render rays, run in optimisation loop
        Returns:
          List of:
            rgb_map: [batch_size, 3]. Predicted RGB values for rays.
            disp_map: [batch_size]. Disparity map. Inverse of depth.
            acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
          Dict of extras: dict with everything returned by render_rays().
        """

        # Render and reshape
        ray_shape = flat_rays.shape  # num_rays, 11

        # assert ray_shape[0] == self.n_rays  # this is not satisfied in test model
        fn = self.volumetric_rendering
        all_ret = batchify_rays(fn, flat_rays, self.chunk)

        for k in all_ret:
            k_sh = list(ray_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        return all_ret

    def render_rays_chunk(self, flat_rays, chunk_size=1024 * 4):
        """Render rays while moving resulting chunks to cpu to avoid OOM when rendering large images."""
        B = flat_rays.shape[0]  # num_rays
        results = defaultdict(list)
        for i in range(0, B, chunk_size):
            rendered_ray_chunks = self.render_rays(flat_rays[i : i + chunk_size])

            for k, v in rendered_ray_chunks.items():
                results[k] += [v.cpu()]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def volumetric_rendering(self, ray_batch):
        """
        Volumetric Rendering
        """
        N_rays = ray_batch.shape[0]

        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [N_rays, 1], [N_rays, 1]

        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples).cuda()

        z_vals = near * (1.0 - t_vals) + far * (
            t_vals
        )  # use linear sampling in depth space

        z_vals = z_vals.expand([N_rays, self.N_samples])

        if self.perturb > 0.0 and self.training:  # perturb sampling depths (z_vals)
            if (
                self.training is True
            ):  # only add perturbation during training intead of testing
                # get intervals between samples
                mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape).cuda()

                z_vals = lower + (upper - lower) * t_rand

        pts_coarse_sampled = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples, 3]

        raw_noise_std = self.raw_noise_std if self.training else 0
        raw_coarse = run_network(
            pts_coarse_sampled,
            viewdirs,
            self.ssr_net_coarse,
            self.embed_fn,
            self.embeddirs_fn,
            netchunk=self.netchunk,
        )
        (
            rgb_coarse,
            disp_coarse,
            acc_coarse,
            weights_coarse,
            depth_coarse,
            sem_logits_coarse,
            feat_map_coarse,
        ) = raw2outputs(
            raw_coarse,
            z_vals,
            rays_d,
            raw_noise_std,
            self.white_bkgd,
            enable_semantic=self.enable_semantic,
            num_sem_class=self.num_valid_semantic_class,
            endpoint_feat=False,
        )

        if self.N_importance > 0:
            z_vals_mid = 0.5 * (
                z_vals[..., 1:] + z_vals[..., :-1]
            )  # (N_rays, N_samples-1) interval mid points
            z_samples = sample_pdf(
                z_vals_mid,
                weights_coarse[..., 1:-1],
                self.N_importance,
                det=(self.perturb == 0.0) or (not self.training),
            )
            z_samples = z_samples.detach()
            # detach so that grad doesn't propogate to weights_coarse from here
            # values are interleaved actually, so maybe can do better than sort?

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts_fine_sampled = (
                rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            )  # [N_rays, N_samples + N_importance, 3]

            raw_fine = run_network(
                pts_fine_sampled,
                viewdirs,
                lambda x: self.ssr_net_fine(x, self.endpoint_feat),
                self.embed_fn,
                self.embeddirs_fn,
                netchunk=self.netchunk,
            )

            (
                rgb_fine,
                disp_fine,
                acc_fine,
                weights_fine,
                depth_fine,
                sem_logits_fine,
                feat_map_fine,
            ) = raw2outputs(
                raw_fine,
                z_vals,
                rays_d,
                raw_noise_std,
                self.white_bkgd,
                enable_semantic=self.enable_semantic,
                num_sem_class=self.num_valid_semantic_class,
                endpoint_feat=self.endpoint_feat,
            )

        ret = {}
        ret["raw_coarse"] = raw_coarse
        ret["rgb_coarse"] = rgb_coarse
        ret["disp_coarse"] = disp_coarse
        ret["acc_coarse"] = acc_coarse
        ret["depth_coarse"] = depth_coarse
        if self.enable_semantic:
            ret["sem_logits_coarse"] = sem_logits_coarse

        if self.N_importance > 0:
            ret["rgb_fine"] = rgb_fine
            ret["disp_fine"] = disp_fine
            ret["acc_fine"] = acc_fine
            ret["depth_fine"] = depth_fine
            if self.enable_semantic:
                ret["sem_logits_fine"] = sem_logits_fine
            ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
            ret["raw_fine"] = raw_fine  # model's raw, unprocessed predictions.
            if self.endpoint_feat:
                ret["feat_map_fine"] = feat_map_fine
        for k in ret:
            if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

    def create_ssr(self):
        """Instantiate NeRF's MLP model."""

        nerf_model = Semantic_NeRF

        embed_fn, input_ch = get_embedder(
            self.config["render"]["multires"],
            self.config["render"]["i_embed"],
            scalar_factor=10,
        )

        input_ch_views = 0
        embeddirs_fn = None
        if self.config["render"]["use_viewdirs"]:
            embeddirs_fn, input_ch_views = get_embedder(
                self.config["render"]["multires_views"],
                self.config["render"]["i_embed"],
                scalar_factor=1,
            )
        output_ch = 5 if self.N_importance > 0 else 4
        skips = [4]
        model = nerf_model(
            enable_semantic=self.enable_semantic,
            num_semantic_classes=self.num_valid_semantic_class,
            D=self.config["model"]["netdepth"],
            W=self.config["model"]["netwidth"],
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=self.config["render"]["use_viewdirs"],
        ).cuda()
        grad_vars = list(model.parameters())

        model_fine = None
        if self.N_importance > 0:
            model_fine = nerf_model(
                enable_semantic=self.enable_semantic,
                num_semantic_classes=self.num_valid_semantic_class,
                D=self.config["model"]["netdepth_fine"],
                W=self.config["model"]["netwidth_fine"],
                input_ch=input_ch,
                output_ch=output_ch,
                skips=skips,
                input_ch_views=input_ch_views,
                use_viewdirs=self.config["render"]["use_viewdirs"],
            ).cuda()
            grad_vars += list(model_fine.parameters())

        # Create optimizer
        optimizer = torch.optim.Adam(params=grad_vars, lr=self.lrate)

        self.ssr_net_coarse = model
        self.ssr_net_fine = model_fine
        self.embed_fn = embed_fn
        self.embeddirs_fn = embeddirs_fn
        self.optimizer = optimizer

    # optimisation step
    def step(self, global_step):
        # Misc
        img2mse = lambda x, y: torch.mean((x - y) ** 2)
        mse2psnr = (
            lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]).cuda())
        )
        CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label - 1)
        to8b_np = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

        # sample rays to query and optimise
        sampled_data = self.sample_data(
            global_step, self.rays, self.H, self.W, no_batching=True, mode="train"
        )
        if self.enable_semantic:
            (
                sampled_rays,
                sampled_gt_rgb,
                _,
                sampled_gt_semantic,
                sematic_available,
            ) = sampled_data
        else:
            sampled_rays, sampled_gt_rgb = sampled_data

        sampled_rays = sampled_rays.cuda()
        sampled_gt_rgb = sampled_gt_rgb.cuda()
        output_dict = self.render_rays(sampled_rays)

        rgb_coarse = output_dict["rgb_coarse"]  # N_rays x 3
        if self.enable_semantic:
            sem_logits_coarse = output_dict["sem_logits_coarse"]  # N_rays x num_classes

        if self.N_importance > 0:
            rgb_fine = output_dict["rgb_fine"]
            if self.enable_semantic:
                sem_logits_fine = output_dict["sem_logits_fine"]

        self.optimizer.zero_grad()

        img_loss_coarse = img2mse(rgb_coarse, sampled_gt_rgb)

        if self.enable_semantic and sematic_available:
            semantic_loss_coarse = crossentropy_loss(
                sem_logits_coarse, sampled_gt_semantic
            )
        else:
            semantic_loss_coarse = torch.tensor(0)

        with torch.no_grad():
            psnr_coarse = mse2psnr(img_loss_coarse)

        if self.N_importance > 0:
            img_loss_fine = img2mse(rgb_fine, sampled_gt_rgb)
            if self.enable_semantic and sematic_available:
                semantic_loss_fine = crossentropy_loss(
                    sem_logits_fine, sampled_gt_semantic
                )
            else:
                semantic_loss_fine = torch.tensor(0)
            with torch.no_grad():
                psnr_fine = mse2psnr(img_loss_fine)
        else:
            img_loss_fine = torch.tensor(0)
            psnr_fine = torch.tensor(0)

        total_img_loss = img_loss_coarse + img_loss_fine
        total_sem_loss = semantic_loss_coarse + semantic_loss_fine

        wgt_sem_loss = float(self.config["train"]["wgt_sem"])
        total_loss = total_img_loss + total_sem_loss * wgt_sem_loss

        total_loss.backward()
        self.optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = self.lrate_decay
        new_lrate = self.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lrate
        ################################

        # tensorboard-logging
        # visualize loss curves
        if global_step % float(self.config["logging"]["step_log_tfb"]) == 0:
            self.tfb_viz.vis_scalars(
                global_step,
                [
                    img_loss_coarse,
                    img_loss_fine,
                    total_img_loss,
                    semantic_loss_coarse,
                    semantic_loss_fine,
                    total_sem_loss,
                    total_sem_loss * wgt_sem_loss,
                    total_loss,
                ],
                [
                    "Train/Loss/img_loss_coarse",
                    "Train/Loss/img_loss_fine",
                    "Train/Loss/total_img_loss",
                    "Train/Loss/semantic_loss_coarse",
                    "Train/Loss/semantic_loss_fine",
                    "Train/Loss/total_sem_loss",
                    "Train/Loss/weighted_total_sem_loss",
                    "Train/Loss/total_loss",
                ],
            )

            # add raw transparancy value into tfb histogram
            trans_coarse = output_dict["raw_coarse"][..., 3]
            self.tfb_viz.vis_histogram(global_step, trans_coarse, "trans_coarse")
            if self.N_importance > 0:
                trans_fine = output_dict["raw_fine"][..., 3]
                self.tfb_viz.vis_histogram(global_step, trans_fine, "trans_fine")

            self.tfb_viz.vis_scalars(
                global_step,
                [psnr_coarse, psnr_fine],
                ["Train/Metric/psnr_coarse", "Train/Metric/psnr_fine"],
            )

        # Rest is logging
        # saving ckpts regularly
        if global_step % float(self.config["logging"]["step_save_ckpt"]) == 0:
            ckpt_dir = os.path.join(self.save_dir, "checkpoints")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            ckpt_file = os.path.join(ckpt_dir, "{:06d}.ckpt".format(global_step))
            torch.save(
                {
                    "global_step": global_step,
                    "network_coarse_state_dict": self.ssr_net_coarse.state_dict(),
                    "network_fine_state_dict": self.ssr_net_fine.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                ckpt_file,
            )
            print("Saved checkpoints at", ckpt_file)

        # render and save training-set images
        if (
            global_step % self.config["logging"]["step_vis_train"] == 0
            and global_step > 0
        ):
            self.training = False  # enable testing mode before rendering results, need to set back during training!
            self.ssr_net_coarse.eval()
            self.ssr_net_fine.eval()
            trainsavedir = os.path.join(
                self.config["experiment"]["save_dir"],
                "train_render",
                "step_{:06d}".format(global_step),
            )
            os.makedirs(trainsavedir, exist_ok=True)
            print(f" {self.num_train} train images")
            with torch.no_grad():
                (
                    rgbs,
                    disps,
                    deps,
                    vis_deps,
                    sems,
                    vis_sems,
                    sem_uncers,
                    vis_sem_uncers,
                ) = self.render_path(self.rays_vis.cuda(), save_dir=trainsavedir)
                #  numpy array of shape [B,H,W,C] or [B,H,W]
            print("Saved training set")

            self.training = True  # set training flag back after rendering images
            self.ssr_net_coarse.train()
            self.ssr_net_fine.train()

            with torch.no_grad():
                if self.enable_semantic:
                    # mask out void regions for better visualisation
                    for idx in range(vis_sems.shape[0]):
                        vis_sems[idx][
                            self.train_semantic_clean_scaled[idx] == self.ignore_label,
                            :,
                        ] = 0

                batch_train_img_mse = img2mse(
                    torch.from_numpy(rgbs), self.train_image_scaled.cpu()
                )
                batch_train_img_psnr = mse2psnr(batch_train_img_mse)

                self.tfb_viz.vis_scalars(
                    global_step,
                    [batch_train_img_psnr, batch_train_img_mse],
                    ["Train/Metric/batch_PSNR", "Train/Metric/batch_MSE"],
                )

            imageio.mimwrite(
                os.path.join(trainsavedir, "rgb.mp4"), to8b_np(rgbs), fps=30, quality=8
            )
            imageio.mimwrite(
                os.path.join(trainsavedir, "dep.mp4"), vis_deps, fps=30, quality=8
            )
            imageio.mimwrite(
                os.path.join(trainsavedir, "disps.mp4"),
                to8b_np(disps / np.max(disps)),
                fps=30,
                quality=8,
            )
            if self.enable_semantic:
                imageio.mimwrite(
                    os.path.join(trainsavedir, "sem.mp4"), vis_sems, fps=30, quality=8
                )
                imageio.mimwrite(
                    os.path.join(trainsavedir, "sem_uncertainty.mp4"),
                    vis_sem_uncers,
                    fps=30,
                    quality=8,
                )

            # add rendered image into tf-board
            self.tfb_viz.tb_writer.add_image(
                "Train/rgb", rgbs, global_step, dataformats="NHWC"
            )
            self.tfb_viz.tb_writer.add_image(
                "Train/depth", vis_deps, global_step, dataformats="NHWC"
            )
            self.tfb_viz.tb_writer.add_image(
                "Train/disps",
                np.expand_dims(disps, -1),
                global_step,
                dataformats="NHWC",
            )

            # evaluate depths
            depth_metrics_dic = calculate_depth_metrics(
                depth_trgt=self.train_depth_scaled, depth_pred=deps
            )
            self.tfb_viz.vis_scalars(
                global_step,
                list(depth_metrics_dic.values()),
                ["Train/Metric/" + k for k in list(depth_metrics_dic.keys())],
            )

            if self.enable_semantic:
                self.tfb_viz.tb_writer.add_image(
                    "Train/vis_sem_label", vis_sems, global_step, dataformats="NHWC"
                )
                self.tfb_viz.tb_writer.add_image(
                    "Train/vis_sem_uncertainty",
                    vis_sem_uncers,
                    global_step,
                    dataformats="NHWC",
                )

                # add segmentation quantative metrics during training into tfb
                (
                    miou_train,
                    miou_train_validclass,
                    total_accuracy_train,
                    class_average_accuracy_train,
                    _,
                ) = calculate_segmentation_metrics(
                    true_labels=self.train_semantic_clean_scaled,
                    predicted_labels=sems,
                    number_classes=self.num_valid_semantic_class,
                    ignore_label=self.ignore_label,
                )
                self.tfb_viz.vis_scalars(
                    global_step,
                    [
                        miou_train,
                        miou_train_validclass,
                        total_accuracy_train,
                        class_average_accuracy_train,
                    ],
                    [
                        "Train/Metric/mIoU",
                        "Train/Metric/mIoU_validclass",
                        "Train/Metric/total_acc",
                        "Train/Metric/avg_acc",
                    ],
                )

                tqdm.write(
                    f"[Training Metric] Iter: {global_step} "
                    f"img_loss: {total_img_loss.item()}, semantic_loss: {(total_sem_loss*wgt_sem_loss).item()},"
                    f"psnr_coarse: {psnr_coarse.item()}, psnr_fine: {psnr_fine.item()},"
                    f"mIoU: {miou_train}, total_acc: {total_accuracy_train}, avg_acc: {class_average_accuracy_train}"
                )

                # Re-compute same metrics for all classes except 'background'
                true_labels_without_background = copy.deepcopy(
                    self.train_semantic_clean_scaled
                )
                true_labels_without_background[
                    true_labels_without_background == 1
                ] = self.ignore_label

                (
                    miou_train,
                    miou_train_validclass,
                    total_accuracy_train,
                    class_average_accuracy_train,
                    _,
                ) = calculate_segmentation_metrics(
                    true_labels=true_labels_without_background,
                    predicted_labels=sems,
                    number_classes=self.num_valid_semantic_class,
                    ignore_label=self.ignore_label,
                )
                self.tfb_viz.vis_scalars(
                    global_step,
                    [
                        miou_train,
                        miou_train_validclass,
                        total_accuracy_train,
                        class_average_accuracy_train,
                    ],
                    [
                        "Train/Metric/mIoU_no_background",
                        "Train/Metric/mIoU_validclass_no_background",
                        "Train/Metric/total_acc_no_background",
                        "Train/Metric/avg_acc_no_background",
                    ],
                )

                # Re-compute same metrics for 'background' class only
                true_labels_only_background = copy.deepcopy(
                    self.train_semantic_clean_scaled
                )
                true_labels_only_background[
                    true_labels_only_background != 1
                ] = self.ignore_label

                (
                    miou_train,
                    miou_train_validclass,
                    total_accuracy_train,
                    class_average_accuracy_train,
                    _,
                ) = calculate_segmentation_metrics(
                    true_labels=true_labels_only_background,
                    predicted_labels=sems,
                    number_classes=self.num_valid_semantic_class,
                    ignore_label=self.ignore_label,
                )
                self.tfb_viz.vis_scalars(
                    global_step,
                    [
                        miou_train,
                        miou_train_validclass,
                        total_accuracy_train,
                        class_average_accuracy_train,
                    ],
                    [
                        "Train/Metric/mIoU_background_only",
                        "Train/Metric/mIoU_validclass_background_only",
                        "Train/Metric/total_acc_background_only",
                        "Train/Metric/avg_acc_background_only",
                    ],
                )

        # render and save test images, corresponding videos
        if global_step % self.config["logging"]["step_val"] == 0 and global_step > 0:
            self.training = False  # enable testing mode before rendering results, need to set back during training!
            self.ssr_net_coarse.eval()
            self.ssr_net_fine.eval()
            testsavedir = os.path.join(
                self.config["experiment"]["save_dir"],
                "test_render",
                "step_{:06d}".format(global_step),
            )
            os.makedirs(testsavedir, exist_ok=True)
            print(f" {self.num_test} test images")
            with torch.no_grad():
                # rgbs, disps, deps, vis_deps, sems, vis_sems, sem_uncers, vis_sem_uncers = self.render_path(self.rays_test, save_dir=testsavedir)
                (
                    rgbs,
                    disps,
                    deps,
                    vis_deps,
                    sems,
                    vis_sems,
                    _,
                    vis_sem_uncers,
                ) = self.render_path(self.rays_test.cuda(), save_dir=testsavedir)
            print("Saved test set")

            self.training = True  # set training flag back after rendering images
            self.ssr_net_coarse.train()
            self.ssr_net_fine.train()

            # Saving test metrics to json file
            test_metrics = {}

            with torch.no_grad():
                if self.enable_semantic:
                    # mask out void regions for better visualisation
                    for idx in range(vis_sems.shape[0]):
                        vis_sems[idx][
                            self.test_semantic_scaled[idx] == self.ignore_label, :
                        ] = 0

                batch_test_img_mse = img2mse(
                    torch.from_numpy(rgbs), self.test_image_scaled.cpu()
                )
                batch_test_img_psnr = mse2psnr(batch_test_img_mse)

                self.tfb_viz.vis_scalars(
                    global_step,
                    [batch_test_img_psnr, batch_test_img_mse],
                    ["Test/Metric/batch_PSNR", "Test/Metric/batch_MSE"],
                )

                # Saving test metrics to json file
                test_metrics["batch_PSNR"] = batch_test_img_psnr.item()
                test_metrics["batch_MSE"] = batch_test_img_mse.item()

            imageio.mimwrite(
                os.path.join(testsavedir, "rgb.mp4"), to8b_np(rgbs), fps=30, quality=8
            )
            imageio.mimwrite(
                os.path.join(testsavedir, "dep.mp4"), vis_deps, fps=30, quality=8
            )
            imageio.mimwrite(
                os.path.join(testsavedir, "disps.mp4"),
                to8b_np(disps / np.max(disps)),
                fps=30,
                quality=8,
            )
            if self.enable_semantic:
                imageio.mimwrite(
                    os.path.join(testsavedir, "sem.mp4"), vis_sems, fps=30, quality=8
                )
                imageio.mimwrite(
                    os.path.join(testsavedir, "sem_uncertainty.mp4"),
                    vis_sem_uncers,
                    fps=30,
                    quality=8,
                )

            # add rendered image into tf-board
            self.tfb_viz.tb_writer.add_image(
                "Test/rgb", rgbs, global_step, dataformats="NHWC"
            )
            self.tfb_viz.tb_writer.add_image(
                "Test/depth", vis_deps, global_step, dataformats="NHWC"
            )
            self.tfb_viz.tb_writer.add_image(
                "Test/disps", np.expand_dims(disps, -1), global_step, dataformats="NHWC"
            )

            # evaluate depths
            depth_metrics_dic = calculate_depth_metrics(
                depth_trgt=self.test_depth_scaled, depth_pred=deps
            )
            self.tfb_viz.vis_scalars(
                global_step,
                list(depth_metrics_dic.values()),
                ["Test/Metric/" + k for k in list(depth_metrics_dic.keys())],
            )

            # Saving test metrics to json file
            for k in depth_metrics_dic.keys():
                test_metrics[k] = str(depth_metrics_dic[k])

            if self.enable_semantic:
                self.tfb_viz.tb_writer.add_image(
                    "Test/vis_sem_label", vis_sems, global_step, dataformats="NHWC"
                )
                self.tfb_viz.tb_writer.add_image(
                    "Test/vis_sem_uncertainty",
                    vis_sem_uncers,
                    global_step,
                    dataformats="NHWC",
                )

                # add segmentation quantative metrics during testung into tfb
                (
                    miou_test,
                    miou_test_validclass,
                    total_accuracy_test,
                    class_average_accuracy_test,
                    _,
                ) = calculate_segmentation_metrics(
                    true_labels=self.test_semantic_scaled,
                    predicted_labels=sems,
                    number_classes=self.num_valid_semantic_class,
                    ignore_label=self.ignore_label,
                )
                # number_classes=self.num_semantic_class-1 to exclude void class
                self.tfb_viz.vis_scalars(
                    global_step,
                    [
                        miou_test,
                        miou_test_validclass,
                        total_accuracy_test,
                        class_average_accuracy_test,
                    ],
                    [
                        "Test/Metric/mIoU",
                        "Test/Metric/mIoU_validclass",
                        "Test/Metric/total_acc",
                        "Test/Metric/avg_acc",
                    ],
                )

                # Saving test metrics to json file
                test_metrics["mIoU"] = miou_test.item()
                test_metrics["mIoU_validclass"] = miou_test_validclass.item()
                test_metrics["total_acc"] = total_accuracy_test.item()
                test_metrics["avg_acc"] = class_average_accuracy_test.item()

                # Re-compute same metrics for all classes except 'background'
                true_labels_without_background = copy.deepcopy(
                    self.test_semantic_scaled
                )
                true_labels_without_background[
                    true_labels_without_background == 1
                ] = self.ignore_label

                (
                    miou_test,
                    miou_test_validclass,
                    total_accuracy_test,
                    class_average_accuracy_test,
                    _,
                ) = calculate_segmentation_metrics(
                    true_labels=true_labels_without_background,
                    predicted_labels=sems,
                    number_classes=self.num_valid_semantic_class,
                    ignore_label=self.ignore_label,
                )
                # number_classes=self.num_semantic_class-1 to exclude void class
                self.tfb_viz.vis_scalars(
                    global_step,
                    [
                        miou_test,
                        miou_test_validclass,
                        total_accuracy_test,
                        class_average_accuracy_test,
                    ],
                    [
                        "Test/Metric/mIoU_no_background",
                        "Test/Metric/mIoU_validclass_no_background",
                        "Test/Metric/total_acc_no_background",
                        "Test/Metric/avg_acc_no_background",
                    ],
                )

                # Saving test metrics to json file
                test_metrics["mIoU_no_background"] = miou_test.item()
                test_metrics[
                    "mIoU_validclass_no_background"
                ] = miou_test_validclass.item()
                test_metrics["total_acc_no_background"] = total_accuracy_test.item()
                test_metrics[
                    "avg_acc_no_background"
                ] = class_average_accuracy_test.item()

                # Re-compute same metrics for 'background' class only
                true_labels_only_background = copy.deepcopy(self.test_semantic_scaled)
                true_labels_only_background[
                    true_labels_only_background != 1
                ] = self.ignore_label

                (
                    miou_test,
                    miou_test_validclass,
                    total_accuracy_test,
                    class_average_accuracy_test,
                    _,
                ) = calculate_segmentation_metrics(
                    true_labels=true_labels_only_background,
                    predicted_labels=sems,
                    number_classes=self.num_valid_semantic_class,
                    ignore_label=self.ignore_label,
                )
                # number_classes=self.num_semantic_class-1 to exclude void class
                self.tfb_viz.vis_scalars(
                    global_step,
                    [
                        miou_test,
                        miou_test_validclass,
                        total_accuracy_test,
                        class_average_accuracy_test,
                    ],
                    [
                        "Test/Metric/mIoU_background_only",
                        "Test/Metric/mIoU_validclass_background_only",
                        "Test/Metric/total_acc_background_only",
                        "Test/Metric/avg_acc_background_only",
                    ],
                )

                # Saving test metrics to json file
                test_metrics["mIoU_background_only"] = miou_test.item()
                test_metrics[
                    "mIoU_validclass_background_only"
                ] = miou_test_validclass.item()
                test_metrics["total_acc_background_only"] = total_accuracy_test.item()
                test_metrics[
                    "avg_acc_background_only"
                ] = class_average_accuracy_test.item()

                # Saving test metrics to json file
                with open(
                    os.path.join(
                        self.config["experiment"]["save_dir"],
                        "test_metrics_step_{:06d}.json".format(global_step),
                    ),
                    "w",
                ) as outfile:
                    json.dump(test_metrics, outfile)

        if global_step % self.config["logging"]["step_log_print"] == 0:
            tqdm.write(
                f"[TRAIN] Iter: {global_step} "
                f"Loss: {total_loss.item()} "
                f"rgb_total_loss: {total_img_loss.item()}, rgb_coarse: {img_loss_coarse.item()}, rgb_fine: {img_loss_fine.item()}, "
                f"sem_total_loss: {total_sem_loss.item()}, weighted_sem_total_loss: {total_sem_loss.item()*wgt_sem_loss}, "
                f"semantic_loss: {semantic_loss_coarse.item()}, semantic_fine: {semantic_loss_fine.item()}, "
                f"PSNR_coarse: {psnr_coarse.item()}, PSNR_fine: {psnr_fine.item()}"
            )

    def render_path(self, rays, save_dir=None):

        rgbs = []
        disps = []

        deps = []
        vis_deps = []

        sems = []
        vis_sems = []

        entropys = []
        vis_entropys = []

        to8b_np = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
        logits_2_label = lambda x: torch.argmax(
            torch.nn.functional.softmax(x, dim=-1), dim=-1
        )
        logits_2_uncertainty = lambda x: torch.sum(
            -F.log_softmax(x, dim=-1) * F.softmax(x, dim=-1), dim=-1, keepdim=True
        )

        t = time.time()
        for i in tqdm(range(len(rays))):
            print(i, time.time() - t)
            t = time.time()
            output_dict = self.render_rays_chunk(
                rays[i], self.chunk
            )  # render rays by chunks to avoid OOM
            rgb_coarse = output_dict["rgb_coarse"]
            disp_coarse = output_dict["disp_coarse"]
            depth_coarse = output_dict["depth_coarse"]

            rgb = rgb_coarse
            depth = depth_coarse
            disp = disp_coarse

            if self.enable_semantic:
                sem_label_coarse = logits_2_label(output_dict["sem_logits_coarse"])
                sem_uncertainty_coarse = logits_2_uncertainty(
                    output_dict["sem_logits_coarse"]
                )
                vis_sem_label_coarse = self.valid_colour_map[sem_label_coarse]
                # shift pred label by +1 to shift the first valid class to use the first valid colour-map instead of the void colour-map
                sem_label = sem_label_coarse
                vis_sem = vis_sem_label_coarse
                sem_uncertainty = sem_uncertainty_coarse

            if self.N_importance > 0:
                rgb_fine = output_dict["rgb_fine"]
                depth_fine = output_dict["depth_fine"]
                disp_fine = output_dict["disp_fine"]

                rgb = rgb_fine
                depth = depth_fine
                disp = disp_fine

                if self.enable_semantic:
                    sem_label_fine = logits_2_label(output_dict["sem_logits_fine"])
                    sem_uncertainty_fine = logits_2_uncertainty(
                        output_dict["sem_logits_fine"]
                    )
                    vis_sem_label_fine = self.valid_colour_map[sem_label_fine]
                    # shift pred label by +1 to shift the first valid class to use the first valid colour-map instead of the void colour-map
                    sem_label = sem_label_fine
                    vis_sem = vis_sem_label_fine
                    sem_uncertainty = sem_uncertainty_fine

            rgb = rgb.cpu().numpy().reshape((self.H_scaled, self.W_scaled, 3))
            depth = depth.cpu().numpy().reshape((self.H_scaled, self.W_scaled))
            disp = disp.cpu().numpy().reshape((self.H_scaled, self.W_scaled))

            rgbs.append(rgb)
            disps.append(disp)

            deps.append(depth)  # save depth in mm
            vis_deps.append(depth2rgb(depth, min_value=self.near, max_value=self.far))

            if self.enable_semantic:
                sem_label = (
                    sem_label.cpu()
                    .numpy()
                    .astype(np.uint8)
                    .reshape((self.H_scaled, self.W_scaled))
                )
                vis_sem = (
                    vis_sem.cpu()
                    .numpy()
                    .astype(np.uint8)
                    .reshape((self.H_scaled, self.W_scaled, 3))
                )
                sem_uncertainty = (
                    sem_uncertainty.cpu()
                    .numpy()
                    .reshape((self.H_scaled, self.W_scaled))
                )
                vis_sem_uncertainty = depth2rgb(sem_uncertainty)

                sems.append(sem_label)
                vis_sems.append(vis_sem)

                entropys.append(sem_uncertainty)
                vis_entropys.append(vis_sem_uncertainty)

            if i == 0:
                print(rgb.shape, disp.shape)

            if save_dir is not None:
                assert os.path.exists(save_dir)
                rgb8 = to8b_np(rgbs[-1])
                disp = disps[-1].astype(np.uint16)
                dep_mm = (deps[-1] * 1000).astype(np.uint16)
                vis_dep = vis_deps[-1]

                rgb_filename = os.path.join(save_dir, "rgb_{:03d}.png".format(i))
                disp_filename = os.path.join(save_dir, "disp_{:03d}.png".format(i))

                depth_filename = os.path.join(save_dir, "depth_{:03d}.png".format(i))
                vis_depth_filename = os.path.join(
                    save_dir, "vis_depth_{:03d}.png".format(i)
                )

                imageio.imwrite(rgb_filename, rgb8)
                imageio.imwrite(disp_filename, disp, format="png", prefer_uint8=False)

                imageio.imwrite(
                    depth_filename, dep_mm, format="png", prefer_uint8=False
                )
                imageio.imwrite(vis_depth_filename, vis_dep)

                if self.enable_semantic:

                    label_filename = os.path.join(
                        save_dir, "label_{:03d}.png".format(i)
                    )
                    vis_label_filename = os.path.join(
                        save_dir, "vis_label_{:03d}.png".format(i)
                    )

                    entropy_filename = os.path.join(
                        save_dir, "entropy_{:03d}.png".format(i)
                    )
                    vis_entropy_filename = os.path.join(
                        save_dir, "vis_entropy_{:03d}.png".format(i)
                    )
                    sem = sems[-1]
                    vis_sem = vis_sems[-1]

                    sem_uncer = to8b_np(entropys[-1])
                    vis_sem_uncer = vis_entropys[-1]

                    imageio.imwrite(label_filename, sem)
                    imageio.imwrite(vis_label_filename, vis_sem)

                    imageio.imwrite(label_filename, sem)
                    imageio.imwrite(vis_label_filename, vis_sem)

                    imageio.imwrite(entropy_filename, sem_uncer)
                    imageio.imwrite(vis_entropy_filename, vis_sem_uncer)

        rgbs = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)
        deps = np.stack(deps, 0)
        vis_deps = np.stack(vis_deps, 0)

        if self.enable_semantic:
            sems = np.stack(sems, 0)
            vis_sems = np.stack(vis_sems, 0)
            entropys = np.stack(entropys, 0)
            vis_entropys = np.stack(vis_entropys, 0)
        else:
            sems = None
            vis_sems = None
            entropys = None
            vis_entropys = None
        return rgbs, disps, deps, vis_deps, sems, vis_sems, entropys, vis_entropys
