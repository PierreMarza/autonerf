################################################################
# Code adapted from https://github.com/Harry-Zhi/semantic_nerf #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)          #
################################################################

import argparse
import time
from tqdm import trange
import yaml

from SSR.datasets import gibson_autonerf_datasets
from SSR.training import trainer


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="SSR/configs/autonerf_config.yaml",
        help="config file name.",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default="", help="Path to training data"
    )
    parser.add_argument(
        "--test_data_dir", type=str, default="", help="Path to test data"
    )
    parser.add_argument(
        "--use_GT_sem",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to use GT semantic masks (or Mask RCNN masks)",
    )

    args = parser.parse_args()
    # Read YAML file
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    trainer.select_gpus(config["experiment"]["gpu"])
    config["experiment"].update(vars(args))

    # Training hyperparameters
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    use_GT_sem = args.use_GT_sem
    use_GT_sem = use_GT_sem == 1

    # Trainer and dataloader
    ssr_trainer = trainer.SSRTrainer(config)
    gibson_data_loader = gibson_autonerf_datasets.GibsonAutoNeRFDatasetCache(
        train_data_dir=train_data_dir,
        test_data_dir=test_data_dir,
        img_h=config["experiment"]["height"],
        img_w=config["experiment"]["width"],
        use_GT_sem=use_GT_sem,
    )

    ssr_trainer.set_params_autonerf()
    ssr_trainer.prepare_data_autonerf(gibson_data_loader)

    # Create nerf model, init optimizer
    ssr_trainer.create_ssr()
    # Create rays in world coordinates
    ssr_trainer.init_rays()

    start = 0
    N_iters = int(float(config["train"]["N_iters"])) + 1
    global_step = start
    # Training loop
    for i in trange(start, N_iters):
        time0 = time.time()
        ssr_trainer.step(global_step)

        dt = time.time() - time0
        print()
        print("Time per step is :", dt)
        global_step += 1


if __name__ == "__main__":
    train()
