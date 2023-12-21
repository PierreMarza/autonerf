######################################################################
# Code adapted from https://github.com/nerfstudio-project/nerfstudio #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                #
######################################################################

from __future__ import annotations

from typing import Dict

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from autonerfacto.autonerfacto_datamanager import AutonerfactoDataManagerConfig
from autonerfacto.autonerfacto_dataparser import AutonerfactoDataParserConfig
from autonerfacto.autonerfacto import AutonerfactoModelConfig

method_configs: Dict[str, TrainerConfig] = {}
descriptions = {
    "auto-nerfacto": "Inspired from nerfacto. Differences: (i) trained and evaluated on different sets, (ii) semantic head, (iii) additional metrics.",
}

method_configs["autonerfacto"] = TrainerConfig(
    method_name="autonerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    steps_per_eval_all_images=30000,
    max_num_iterations=30001,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=AutonerfactoDataManagerConfig(
            dataparser=AutonerfactoDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="off",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
            ),
        ),
        model=AutonerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

AnnotatedBaseConfigUnion = (
    tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
        tyro.conf.FlagConversionOff[
            tyro.extras.subcommand_type_from_defaults(
                defaults=method_configs, descriptions=descriptions
            )
        ]
    ]
)
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
