######################################################################
# Code adapted from https://github.com/nerfstudio-project/nerfstudio #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                #
######################################################################

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers import base_datamanager


from autonerfacto.autonerfacto_dataset import AutonerfactoDataset


@dataclass
class AutonerfactoDataManagerConfig(base_datamanager.VanillaDataManagerConfig):
    """Autonerfacto datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: AutonerfactoDataManager)


class AutonerfactoDataManager(
    base_datamanager.VanillaDataManager
):  # pylint: disable=abstract-method
    """Data manager implementation for data that also requires processing sem data.
    Args:
        config: the DataManagerConfig used to instantiate class
    """

    def create_train_dataset(self) -> AutonerfactoDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(
            split="train"
        )
        return AutonerfactoDataset(
            dataparser_outputs=self.train_dataparser_outputs,
        )

    def create_eval_dataset(self) -> AutonerfactoDataset:
        return AutonerfactoDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(
                split=self.test_split
            ),
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)

        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        batch = self.eval_pixel_sampler.sample(image_batch)

        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch
