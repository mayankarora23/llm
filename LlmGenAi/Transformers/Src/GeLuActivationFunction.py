#######################################################################
# File - GeLuActivationFunction.py
# Author - Mayank Arora
# Description - This file contains the implementation of GeLu
#               activation function.
#######################################################################

import sys
import torch
import torch.nn as nn

sys.path.append("../../Config/Src")
sys.path.append("../../../Common")

from Logging import Logging
from Logging import LogLevel
from Config import Config

class GeLuActivationFunction(nn.Module):
    def __init__(self, config: Config, device: str = "cpu"):
        super().__init__()
        self.config = config

        Logging.GetInstance().Debug("GeLuActivationFunction initialized with config: {}".format(config))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        Logging.GetInstance().Debug(f"GeLuActivationFunction forward called with inputs: {inputs}")

        # GeLU activation function: x * P(X <= x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        geluOutput = 0.5 * inputs * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2 / torch.pi)) *
            (inputs + 0.044715 * torch.pow(inputs,3))
            ))

        Logging.GetInstance().Debug(f"GeLuActivationFunction output: {geluOutput}")
        return geluOutput
