import numpy as np
from typing import Callable, Sequence

import flax.linen as nn


class MLP(nn.Module):
    layers: Sequence[int]
    hidden_activation: Callable = nn.relu
    output_activation: Callable = lambda s: s
            
    @nn.compact
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nn.Dense(layer)(x)
            x = self.hidden_activation(x)
        x = nn.Dense(self.layers[-1])(x)
        x = self.output_activation(x)
        return x

    
class MLPWithBatchNorm(nn.Module):
    layers: Sequence[int]
    hidden_activation: Callable = nn.relu
    output_activation: Callable = lambda s: s
            
    @nn.compact
    def __call__(self, x, train:bool):
        for layer in self.layers[:-1]:
            x = nn.Dense(layer)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = self.hidden_activation(x)
        x = nn.Dense(self.layers[-1])(x)
        x = self.output_activation(x)
        return x

