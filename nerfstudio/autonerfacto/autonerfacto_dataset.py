######################################################################
# Code adapted from https://github.com/nerfstudio-project/nerfstudio #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                #
######################################################################

from torchtyping import TensorType
from typing import Dict

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path

from autonerfacto.autonerfacto_utils import get_auto_sem_mask_from_path


class AutonerfactoDataset(InputDataset):
    """Autonerfacto dataset that returns images and sem.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(
        self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0
    ):
        super().__init__(dataparser_outputs, scale_factor)
        assert (
            "gt_sem_filenames" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["gt_sem_filenames"] is not None
        )
        self.gt_sem_filenames = self.metadata["gt_sem_filenames"]

        if (
            "depth_filenames" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["depth_filenames"] is not None
        ):
            self.depth_filenames = self.metadata["depth_filenames"]
            self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]
        else:
            self.depth_filenames = None

    def get_embs(self) -> TensorType[...]:
        return self.embs

    def get_metadata(self, data: Dict) -> Dict:
        # Sem
        sem_filepath = self.gt_sem_filenames[data["image_idx"]]
        semantics = get_auto_sem_mask_from_path(filepath=sem_filepath)

        # Depth
        if self.depth_filenames is not None:
            depth_filename = self.depth_filenames[data["image_idx"]]

            height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
            width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

            # Scale depth images to meter units and also by scaling applied to cameras
            scale_factor = (
                self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
            )
            depth = get_depth_image_from_path(
                filepath=depth_filename,
                height=height,
                width=width,
                scale_factor=scale_factor,
            )
        else:
            depth = None

        if depth is not None:
            return {"semantics": semantics, "depth": depth}
        else:
            return {"semantics": semantics}
