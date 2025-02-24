from typing import Optional, Union

import torch.nn as nn
from .layerMKM import *

__all__ = ["init_normal"]


def init_normal(
    module: Union[nn.Linear, nn.Embedding, CustomLayerMKM, FeedForwardProjMKM, AttentionProjMKM],
    std: float,
    init_cutoff_factor: Optional[float] = None,
):
    """
    Initializes the weights and biases of a module using a normal distribution or
    truncated normal distribution if `init_cutoff_factor` is provided.

    Args:
        module: The module to initialize. Can be nn.Linear, nn.Embedding, or CustomLayerMKM.
        std: Standard deviation of the normal distribution.
        init_cutoff_factor: Factor to determine truncation range for truncated normal initialization.
    """
    if isinstance(module, CustomLayerMKM) or isinstance(module, FeedForwardProjMKM) or\
       isinstance(module, AttentionProjMKM):
        # Initialize weights in CustomLayerMKM's expansions
        for w in module.expansions:
            if init_cutoff_factor is not None:
                cutoff_value = init_cutoff_factor * std
                nn.init.trunc_normal_(w, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
            else:
                nn.init.normal_(w, mean=0.0, std=std)

    else:
        # Initialize weights for other modules
        if init_cutoff_factor is not None:
            cutoff_value = init_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # Initialize biases
    if (isinstance(module, nn.Linear) or hasattr(module, 'bias')) and module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, CustomLayerMKM) and module.bias_O is not None:
        nn.init.zeros_(module.bias_O)