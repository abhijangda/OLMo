import torch
from torch import nn
import math

"""
Dimension key:
B: batch size
I1: input feature dimension 1
I2: input feature dimension 2
O1: output feature dimension 1
O2: output feature dimension 2
I: total input features (I = I1 * I2)
O: total output features (O = O1 * O2)
"""

class CustomLayerMKM(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        factors,  # List of dims for 2-D Kronecker factor 
        mkm_type: str = "multi",  # Type: 'multi', 'multi_partial'
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Initializes the CustomLayerMKM_Multi layer.

        Args:
            in_features: Total input size, I = I1 * I2
            out_features: Total output size, O = O1 * O2
            factor_1: List of factor values for the first Kronecker factor (length n_expansions)
            factor_2: List of factor values for the second Kronecker factor (length n_expansions)
            mkm_type: Operation mode ('multi', 'multi_partial')
            bias: Whether to include a bias term
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # Validate `mkm_type`
        valid_types = ["multi", "multi_partial"]
        if mkm_type not in valid_types:
            raise ValueError(f"Invalid mkm_type '{mkm_type}'. Must be one of {valid_types}.")
        self.mkm_type = mkm_type

        # Ensure factor_1 and factor_2 are lists
        if not isinstance(factors, list):
            raise TypeError("factors must be a list.")
            for factor in factors:
                if not isinstance(factor, list) or not len(factor) == 2:
                    raise TypeError("factor must be list of 2 integers.")

        # Ensure that the total input and output features are divisible by the factors
        for f1, f2 in factors:
            if in_features % f1 != 0:
                raise ValueError(f"Input features ({in_features}) must be divisible by factor_1 ({f1}).")
            if out_features % f2 != 0:
                raise ValueError(f"Output features ({out_features}) must be divisible by factor_2 ({f2}).")

        self._in_features = in_features   # Total input features (I)
        self._out_features = out_features # Total output features (O)
        self.n_expansions = len(factors) # Number of expansions

        # Store expansions as pairs of smaller weight matrices
        self.expansions = nn.ParameterList()
        self.expansions_shape = []

        for idx, (f1, f2) in enumerate(factors):
            # Decompose dimensions for Kronecker factors
            # Input dimensions I = I1 * I2, where I1 = in_features_I1, I2 = in_features_I2
            in_features_I1 = in_features // f1  # Compute I1
            in_features_I2 = f1                 # Set I2

            # Output dimensions O = O1 * O2, where O1 = out_features_O1, O2 = out_features_O2
            out_features_O1 = out_features // f2  # Compute O1
            out_features_O2 = f2                  # Set O2

            # Initialize weight matrices for each Kronecker factor
            # weight_O1I1: Shape [O1, I1]
            weight_O1I1 = nn.Parameter(torch.zeros(
                (out_features_O1, in_features_I1), **factory_kwargs
            ))
            # weight_O2I2: Shape [O2, I2]
            weight_O2I2 = nn.Parameter(torch.zeros(
                (out_features_O2, in_features_I2), **factory_kwargs
            ))

            # Append weight matrices to the ParameterList
            self.expansions.append(weight_O1I1)
            self.expansions.append(weight_O2I2)
            self.expansions_shape += [weight_O1I1.shape, weight_O2I2.shape]

        
        # Initialize bias term
        if bias:
            # bias_O: Shape [O]
            self.bias_O = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias_O", None)

        self.reset_parameters()

    @property
    def in_features(self):
        return self._in_features  # Total input features (I)

    @property
    def out_features(self):
        return self._out_features  # Total output features (O)

    def reset_parameters(self):
        """
        Initializes the weight matrices and bias term.
        """
        # Initialize all weight matrices using Kaiming uniform initialization
        for weight_O1o2I1o2 in self.expansions:
            nn.init.kaiming_uniform_(weight_O1o2I1o2, a=math.sqrt(5))

        # Initialize bias term if present
        if self.bias_O is not None:
            # Use a uniform distribution U(-bound, bound)
            bound = 1 / math.sqrt(self._in_features)
            nn.init.uniform_(self.bias_O, -bound, bound)
            
    def forward(self, input_BI, expansions_to_use=None):
        """
        Forward pass of the CustomLayerMKM_Multi layer.

        Args:
            input_BI: Input tensor of shape [B, I], where B is batch size, I is total input features
            expansions_to_use: List of expansion indices to use (only for 'multi_partial' mode)

        Returns:
            output_BO: Output tensor of shape [B, O], where O is total output features
        """
        if self.mkm_type == "multi_partial":
            if expansions_to_use is None:
                raise ValueError("expansions_to_use must be specified for 'multi_partial' mode.")
            expansions_indices = expansions_to_use
        else:
            # For 'multi' mode, use all expansions
            expansions_indices = range(self.n_expansions)

        # Accumulate sum of Kronecker products over specified expansions
        W_sum_OI = None
        for i in expansions_indices:
            weight_O1I1 = self.expansions[2 * i]     # [O1, I1]
            weight_O2I2 = self.expansions[2 * i + 1] # [O2, I2]
            # Compute Kronecker product of weight matrices
            # kron_OI: Shape [O1 * O2, I1 * I2] = [O, I]
            kron_OI = torch.kron(weight_O2I2, weight_O1I1)
            # Sum the Kronecker products
            if W_sum_OI is None:
                W_sum_OI = kron_OI
            else:
                W_sum_OI = W_sum_OI + kron_OI

        # Perform the matrix multiplication with input
        # input_BI: Shape [B, I]
        # W_sum_OI.t(): Shape [I, O]
        # output_BO: Shape [B, O]
        output_BO = input_BI.matmul(W_sum_OI.t())

        # Add bias term if present
        if self.bias_O is not None:
            output_BO = output_BO + self.bias_O

        return output_BO

    def extra_repr(self):
        """
        Returns a string representation of the layer, including the sizes of the matrices in `expansions`.
        """
        # Gather sizes of each pair of weight matrices in `expansions`
        expansion_sizes = []
        for i in range(0, len(self.expansions), 2):
            shape1 = self.expansions_shape[i]
            shape2 = self.expansions_shape[i + 1]
            expansion_sizes.append(f"expansion_{i // 2}: {shape1} ⊗ {shape2}")

        # Combine sizes into a single string for readability
        expansion_sizes_str = "\n    ".join(expansion_sizes)

        # Explicitly return as a single string without using parentheses
        return "in_features=" + str(self._in_features) + ", " + \
            "out_features=" + str(self._out_features) + ", " + \
            "mkm_type=" + str(self.mkm_type) + ", " + \
            "n_expansions=" + str(self.n_expansions) + ", " + \
            "bias=" + str(self.bias_O is not None) + "\n" + \
            "Expansions:\n    " + expansion_sizes_str

class FeedForwardProjMKM(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        factors,  # List of dims for 2-D Kronecker factor 
        mkm_type: str = "multi",  # Type: 'multi', 'multi_partial'
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self._in_features = in_features   # Total input features (I)
        self._out_features = out_features # Total output features (O)
        self.mkm_w1 = CustomLayerMKM(in_features, out_features//2, factors, mkm_type, bias, device, dtype)
        self.mkm_w2 = CustomLayerMKM(in_features, out_features//2, factors, mkm_type, bias, device, dtype)
        self.reset_parameters()

    @property
    def in_features(self):
        return self._in_features  # Total input features (I)

    @property
    def out_features(self):
        return self._out_features  # Total output features (O)

    @property
    def expansions(self):
        return list(self.mkm_w1.expansions) + list(self.mkm_w2.expansions)

    def forward(self, x, expansions_to_use=None):
        xw1 = self.mkm_w1(x, expansions_to_use)
        xw2 = self.mkm_w2(x, expansions_to_use)
        return torch.cat((xw1, xw2), dim=-1)
    
    def reset_parameters(self):
        self.mkm_w1.reset_parameters()
        self.mkm_w2.reset_parameters()