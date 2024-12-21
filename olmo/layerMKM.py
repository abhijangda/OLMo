import torch
from torch import nn
import math

from pyfastkron import fastkrontorch as fk

class CustomLayerMKM(nn.Module):
    def __init__(self, in_features, out_features, factor_1, factor_2, bias=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(CustomLayerMKM, self).__init__()
        self.in_features_1 = in_features//factor_1
        self.out_features_1 = out_features//factor_2
        self.in_features_2 = factor_1
        self.out_features_2 = factor_2
        self.device = device

        self.weight_1 = nn.Parameter(torch.zeros((self.out_features_1, self.in_features_1), **factory_kwargs))
        self.weight_2 = nn.Parameter(torch.zeros((self.out_features_2, self.in_features_2), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), **factory_kwargs)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
    @property
    def in_features(self):
        return self.in_features_1 * self.in_features_2
    @property
    def out_features(self):
        return self.out_features_1 * self.out_features_2

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))

        # nn.init.constant_(self.weight_1, 1)
        # nn.init.constant_(self.weight_2, 1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(torch.kron(self.weight_1, self.weight_2))
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def set_use_fk(self, use_fk):
        self.use_fk = use_fk

    def forward(self, input):
        s = input.matmul(torch.kron(self.weight_2, self.weight_1).t())
        # s = fk.gemkm(input, (self.weight_2.mT, self.weight_1.mT))
        if self.bias is not None:
            return s + self.bias
        else:
            return s

    def extra_repr(self) -> str:
        return f"in_features={[self.in_features_1, self.in_features_2]}, out_features={[self.out_features_1, self.out_features_2]}, bias={self.bias is not None}"