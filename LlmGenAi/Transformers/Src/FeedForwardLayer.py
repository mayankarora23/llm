#######################################################################
# File - FeedForwardLayer.py
# Author - Mayank Arora
# Description - This file contains the implementation of feed forward
#               for the transformers.
#######################################################################

import sys


import torch
import torch.nn as nn

sys.path.append("../../../Common")
sys.path.append("../../Config/Src")
from Logging import Logging
from Logging import LogLevel

from Config import Config
from GeLuActivationFunction import GeLuActivationFunction

#######################################################################
# Class - FeedForwardLayer
# Description - This class implements the feed forward layer
#               for the transformers.
#######################################################################
class FeedForwardLayer(nn.Module):
    __config : Config
    __feedForwardLayer : nn.Module
    __linear1 : nn.Linear
    __device : str
    __activation : GeLuActivationFunction
    __linear2 : nn.Linear
    __dropout : nn.Dropout
    def __init__(self, config: Config, device: str = "cpu"):
        super().__init__()
        self.__config = config

        Logging.GetInstance().Debug("FeedForwardLayer initialized with config: {}".format(config))

        self.__linear1 = nn.Linear(self.__config.getEmbeddingDimension(), self.__config.getEmbeddingDimension() * 4)
        self.__activation = GeLuActivationFunction(self.__config)
        #activation = nn.GELU().to(torch.device(self.__device))
        self.__linear2 = nn.Linear(self.__config.getEmbeddingDimension() * 4, self.__config.getEmbeddingDimension())

        self.__dropout = nn.Dropout(self.__config.getDropoutRate())

        Logging.GetInstance().Debug("FeedForwardLayer initialized with layers: linear1, activation, linear2")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        Logging.GetInstance().Debug(f"FeedForwardLayer forward called with inputs: {inputs}")

        #output = self.__feedForwardLayer(inputs)
        output = self.__linear1(inputs)
        output = self.__activation(output)
        output = self.__dropout(output)
        output = self.__linear2(output)
        Logging.GetInstance().Debug(f"FeedForwardLayer output: {output}\n with shape: {output.shape}")

        return output

    def freezeLinear1(self):
        for param in self.__linear1.parameters():
            param.requires_grad = False
        Logging.GetInstance().Info("FeedForwardLayer Linear1 layer frozen.")

    def freezeLinear2(self):
        for param in self.__linear2.parameters():
            param.requires_grad = False
        Logging.GetInstance().Info("FeedForwardLayer Linear2 layer frozen.")

    def freezeDropout(self):
        for param in self.__dropout.parameters():
            param.requires_grad = False
        Logging.GetInstance().Info("FeedForwardLayer Dropout layer frozen.")

    def unfreezeLinear1(self):
        for param in self.__linear1.parameters():
            param.requires_grad = True
        Logging.GetInstance().Info("FeedForwardLayer Linear1 layer unfrozen.")

    def unfreezeLinear2(self):
        for param in self.__linear2.parameters():
            param.requires_grad = True
        Logging.GetInstance().Info("FeedForwardLayer Linear2 layer unfrozen.")

    def unfreezeDropout(self):
        for param in self.__dropout.parameters():
            param.requires_grad = True
        Logging.GetInstance().Info("FeedForwardLayer Dropout layer unfrozen.")
