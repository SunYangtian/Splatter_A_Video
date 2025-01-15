from .dptr import RENDERER_REGISTRY, DPTRRender
from .dptr_ortho import DPTROrthoRender
from .dptr_ortho_enhanced import DPTROrthoEnhancedRender

def parse_renderer(cfg, **kwargs):
    """
    Parse the renderer.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    """
    name = cfg.pop("name")
    if name == 'GaussianSplattingRender':
        from .base_splatting import GaussianSplattingRender
    return RENDERER_REGISTRY.get(name)(cfg, **kwargs)