#######################################################################
# File - LayerNormalization.py
# Author - Mayank Arora
# Description - This file contains the implementation of Layer
#               normalization using PyTorch.
#######################################################################

import sys


import torch
import torch.nn as nn

sys.path.append("../../../Common")
sys.path.append("../../Config/Src")
from Logging import Logging
from Logging import LogLevel


from Config import Config

class LayerNormalization(nn.Module):
    __config: Config
    # parameters scale and shift are trainable parameters with same dimension as the input in forward.
    __scale: torch.Tensor
    __shift: torch.Tensor
    # Small constant added while doing normalization to avoid division by zero error
    __epsilon: float = 1e-5

    def __init__(self, config: Config, device: str = "cpu"):
        super().__init__()
        self.__config = config

        embeddingDimension = self.__config.getEmbeddingDimension()
        self.__scale = nn.Parameter(torch.ones(embeddingDimension))
        self.__shift = nn.Parameter(torch.zeros(embeddingDimension))
        self.__epsilon = 1e-6

        Logging.GetInstance().Debug(f"Initializing LayerNormalization with config: {self.__config.getConfigType()}")

    def __calculateMean(self, inputs: torch.Tensor) -> int:
        mean = inputs.mean(dim=-1, keepdim=True)
        Logging.GetInstance().Debug(f"Calculated mean: {mean}")
        return mean

    def __calculateVariance(self, inputs: torch.Tensor) -> int:
        # For variance calculation we are using unbiased=False, to understand this let's first understand how
        # variance is calculated. So the equation for variance is:
        # variance = sum((x - mean)^2) / N
        # where N is the number of elements in the dimension we are calculating variance over.
        #
        # So when we choose unbiased=True, we do what is called as Bessel's correction, which typically uses
        # N-1 in the denominator instead of N. This is done to correct the bias in the estimation of the population
        # variance from a sample. However, in the context of layer normalization, we are not estimating the
        # population variance from a sample, we are calculating the variance over the entire input tensor.
        # Therefore, we use unbiased=False to calculate the variance over the entire input tensor.
        # This means we are using the formula:
        # variance = sum((x - mean)^2) / N
        variance = inputs.var(dim=-1, unbiased=False, keepdim=True)
        Logging.GetInstance().Debug(f"Calculated variance: {variance}")
        return variance

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = self.__calculateMean(inputs)
        variance = self.__calculateVariance(inputs)
        normalized = (inputs - mean) / torch.sqrt(variance + self.__epsilon)
        output = normalized * self.__scale + self.__shift

        Logging.GetInstance().Debug(f"LayerNormalization output shape: {output.shape}")
        Logging.GetInstance().Debug(f"LayerNormalization output: {output}")
        return output
