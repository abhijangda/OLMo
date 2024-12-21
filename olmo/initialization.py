from typing import Optional, Union

import torch.nn as nn
from .layerMKM import CustomLayerMKM

__all__ = ["init_normal"]


def init_normal(
    module: Union[nn.Linear, nn.Embedding],
    std: float,
    init_cutoff_factor: Optional[float] = None,
):
    # weights
    if init_cutoff_factor is not None:
        cutoff_value = init_cutoff_factor * std
        if type(module) is CustomLayerMKM:
            nn.init.trunc_normal_(module.weight_1, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
            nn.init.trunc_normal_(module.weight_2, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
    else:
        nn.init.normal_(module.weight, mean=0.0, std=std)

    # biases
    # TODO: Change here
    if (isinstance(module, nn.Linear) or hasattr(module, 'bias')) and module.bias is not None:
        nn.init.zeros_(module.bias)
