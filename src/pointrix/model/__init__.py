from .gaussian_points.gaussian_points import GaussianPointCloud
from pointrix.dataset.base_data import BaseDataPipeline
from .base_model import BaseModel, MODEL_REGISTRY

__all__ = ["GaussianPointCloud", "BaseModel"]


def parse_model(cfg, datapipeline:BaseDataPipeline, device="cuda"):
    """
    Parse the model.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    datapipeline : BaseDataPipeline
        The data pipeline.
    device : str
        The device to use.
    """
    name = cfg.pop("name")
    return MODEL_REGISTRY.get(name)(cfg, datapipeline, device)