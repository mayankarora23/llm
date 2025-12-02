#######################################################################
# File - TransformerDecoder.py
# Author - Mayank Arora
# Description - This file contains the implementation of transformer
#               decoder using multi head attention.
#######################################################################

import sys


import torch
import torch.nn as nn

sys.path.append("../../../Common")
sys.path.append("../../Config/Src")
from Logging import Logging
from Logging import LogLevel

from Config import Config
from LayerNormalization import LayerNormalization
from Attention import MaskedMultiHeadAttention
from Attention import CrossMultiHeadAttention
from FeedForwardLayer import FeedForwardLayer

class TransformerDecoder(nn.Module):

    __config                    : Config
    __maskedMultiHeadAttention  : MaskedMultiHeadAttention
    __crossMultiHeadAttention   : CrossMultiHeadAttention
    __layerNorm1                : LayerNormalization
    __layerNorm2                : LayerNormalization
    __layerNorm3                : LayerNormalization
    __feedForwardLayer          : FeedForwardLayer
    __dropout                   : nn.Dropout
    __device                    : str
    __decoderDropoutScaling     : float = 1  # Default dropout scaling for decoder
    __selfAttentionDropout      : nn.Dropout
    __usePreLayerNorm           : bool = True  # Default to pre-layer normalization


    def __init__(self, config : Config, device: str = "cpu"):
        super().__init__()

        self.__config = config

        self.__usePreLayerNorm = self.__config.usePreNormalisation()
        self.__layerNorm1 = LayerNormalization(config)
        self.__maskedMultiHeadAttention = MaskedMultiHeadAttention(config.getEmbeddingDimension(),
                                                       config.getEmbeddingDimension(),
                                                       config.getContextLength(),
                                                       config.getAttentionHeads(),
                                                       config.useQueryKeyValueBias(),
                                                       config.getDropoutRate())

        self.__crossMultiHeadAttention = CrossMultiHeadAttention(config.getEmbeddingDimension(),
                                                       config.getEmbeddingDimension(),
                                                       config.getContextLength(),
                                                       config.getAttentionHeads(),
                                                       config.useQueryKeyValueBias(),
                                                       config.getDropoutRate())

        self.__layerNorm2 = LayerNormalization(config)
        self.__layerNorm3 = LayerNormalization(config)
        self.__feedForwardLayer = FeedForwardLayer(config)
        self.__dropout = nn.Dropout(config.getDropoutRate() * self.__decoderDropoutScaling)
        self.__selfAttentionDropout = nn.Dropout(config.getDropoutRate() * self.__decoderDropoutScaling)
        Logging.GetInstance().Debug(f"TransformerDecoder initialized with config: {self.__config.getConfigType()}")

    def getCrossAttentionWeights(self) -> torch.Tensor:
        if self.__crossMultiHeadAttention is not None:
            return self.__crossMultiHeadAttention.getAttentionWeights()
        else:
            Logging.GetInstance().Error("Cross MultiHeadAttention is not initialized.")
            raise ValueError("Cross MultiHeadAttention is not initialized.")

    def setDropoutScaling(self, dropoutScaling: float):
        self.__selfAttentionDropout = nn.Dropout(self.__config.getDropoutRate() * dropoutScaling)

    def forward(self, inputs: torch.Tensor, encoderOutputs: torch.Tensor = None) -> torch.Tensor:
        # Here it is assumed that inputs received are layer normalized after token and position embedding.
        Logging.GetInstance().Debug(f"inputs shape : {inputs.shape}")

        residual1 = inputs
        if self.__usePreLayerNorm:
            # Multi-head attention layer with pre layer normalization
            inputs = self.__layerNorm1(inputs)

        # Perform masked multi-head self-attention. This is used to ensure that the decoder can only attend to
        # previous tokens in the sequence.
        selfAttentionOutput = self.__maskedMultiHeadAttention(inputs)
        selfAttentionOutput = self.__selfAttentionDropout(selfAttentionOutput)
        attentionOutput = selfAttentionOutput + residual1

        if not self.__usePreLayerNorm:
            # If post layer normalization is used apply normalization here
            attentionOutput = self.__layerNorm1(attentionOutput)
        Logging.GetInstance().Debug(f"outputs after attention : {attentionOutput.shape}")

        if encoderOutputs is not None:
            residual2 = attentionOutput
            if self.__usePreLayerNorm:
                # Multi-head attention layer with pre layer normalization
                attentionOutput = self.__layerNorm2(attentionOutput)

            crossAttentionOutput = self.__crossMultiHeadAttention(attentionOutput, encoderOutputs)
            crossAttentionOutput = self.__dropout(crossAttentionOutput)
            attentionOutput = crossAttentionOutput + residual2
            if not self.__usePreLayerNorm:
                # If post layer normalization is used apply normalization here
                attentionOutput = self.__layerNorm2(attentionOutput)
            Logging.GetInstance().Debug(f"outputs after cross attention : {attentionOutput.shape}")

        residual3 = attentionOutput
        if self.__usePreLayerNorm:
            # Feed forward layer with pre layer normalization
            attentionOutput = self.__layerNorm3(attentionOutput)

        feedForwardOutput = self.__feedForwardLayer(attentionOutput)
        feedForwardOutput = self.__dropout(feedForwardOutput)
        output = feedForwardOutput + residual3

        if not self.__usePreLayerNorm:
            # If post layer normalization is used apply normalization here
            output = self.__layerNorm3(output)
        Logging.GetInstance().Debug(f"outputs after feed forward : {output.shape}")

        return output

    def freezeDropout(self):
        for param in self.__dropout.parameters():
            param.requires_grad = False
        Logging.GetInstance().Debug("Freezing TransformerDecoder dropout parameters.")

    def unfreezeDropout(self):
        for param in self.__dropout.parameters():
            param.requires_grad = True
        Logging.GetInstance().Debug("Unfreezing TransformerDecoder dropout parameters.")

    def freezeSelfAttentionDropout(self):
        for param in self.__selfAttentionDropout.parameters():
            param.requires_grad = False
        Logging.GetInstance().Debug("Freezing TransformerDecoder self-attention dropout parameters.")

    def unfreezeSelfAttentionDropout(self): 
        for param in self.__selfAttentionDropout.parameters():
            param.requires_grad = True
        Logging.GetInstance().Debug("Unfreezing TransformerDecoder self-attention dropout parameters.")

    def getMaskedMultiHeadAttention(self) -> MaskedMultiHeadAttention:
        return self.__maskedMultiHeadAttention

    def getCrossMultiHeadAttention(self) -> CrossMultiHeadAttention:
        return self.__crossMultiHeadAttention

    def getFeedForwardLayer(self) -> FeedForwardLayer:
        return self.__feedForwardLayer

    def getLayerNorm1(self) -> LayerNormalization:
        return self.__layerNorm1

    def getLayerNorm2(self) -> LayerNormalization:
        return self.__layerNorm2

    def getLayerNorm3(self) -> LayerNormalization:
        return self.__layerNorm3
    