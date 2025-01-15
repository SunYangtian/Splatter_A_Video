import numpy as np

from pointrix.utils.registry import Registry
from .optimizer import OptimizerList
SCHEDULER_REGISTRY = Registry("Scheduler", modules=["pointrix.optimizer"])
SCHEDULER_REGISTRY.__doc__ = ""

@SCHEDULER_REGISTRY.register()
class ExponLRScheduler:
    """
    A learning rate scheduler using exponential decay.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    lr_scale : float, optional
        The learning rate scale, by default 1.0
    """

    def __init__(self, config:dict, lr_scale=1.0) -> None:
        scheduler = self.get_expon_lr_func
        self.config = config

        max_steps = config.max_steps
        params = [
            {
                "name": name,
                "init": values["init"] * lr_scale,
                "final": values["final"] * lr_scale,
                "max_steps": max_steps,
            }
            for name, values in config.params.items()
        ]
        self.scheduler_funcs = {}
        for param in params:
            self.scheduler_funcs[param["name"]] = (
                scheduler(
                    init=param["init"],
                    final=param["final"],
                    max_steps=param["max_steps"],
                )
            )

    def get_expon_lr_func(self, init:float, 
                          final:float, 
                          delay_steps:int=0, 
                          delay_mult:float=0.01, 
                          max_steps:int=1000000) -> None:
        """
        Get the exponential learning rate function.

        Parameters
        ----------
        init : float
            The initial learning rate.
        final : float
            The final learning rate.
        delay_steps : int, optional
            The delay steps, by default 0
        delay_mult : float, optional
            The delay multiplier, by default 0.01
        max_steps : int, optional
            The maximum steps, by default 1000000
        """

        def helper(step):
            if step < 0 or (init == 0.0 and final == 0.0):
                # Disable this parameter
                return 0.0
            if delay_steps > 0:
                # A kind of reverse cosine decay.
                delay_rate = delay_mult + (1 - delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / delay_steps, 0, 1)
                )
            else:
                delay_rate = 1.0
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(init) * (1 - t) + np.log(final) * t)
            return delay_rate * log_lerp

        return helper
    
    def step(self, global_step: int, optimizer_list: OptimizerList) -> None:
        """
        Update the learning rate for the optimizer.

        Parameters
        ----------
        global_step : int
            The global step in training.
        optimizer_list : OptimizerList
            The list of all the optimizers which need to be updated.
        """
        for param_group in optimizer_list.param_groups:
                name = param_group['name']
                if name in self.scheduler_funcs.keys():
                    lr = self.scheduler_funcs[name](global_step)
                    param_group['lr'] = lr
        
