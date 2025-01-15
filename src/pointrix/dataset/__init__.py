import os
import sys

from .base_data import DATA_FORMAT_REGISTRY, BaseDataPipeline
from .colmap_data import ColmapReFormat
from .nerf_data import NerfReFormat
from .image_data import ImageReFormat
from .imageDepth_data import ImageDepthReFormat
from .fixCamera_data import FixCameraReFormat


def parse_data_pipeline(cfg: dict):
    """
    Parse the data pipeline.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    """
    if len(cfg) == 0:
        return None
    data_type = cfg.data_type
    dataformat = DATA_FORMAT_REGISTRY.get(data_type)

    return BaseDataPipeline(cfg, dataformat)
