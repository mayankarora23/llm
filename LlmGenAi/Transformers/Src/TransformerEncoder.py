#######################################################################
# File - TransformerEncoder.py
# Author - Mayank Arora
# Description - This file contains the implementation of transformer
#               encoder using multi head attention.
#######################################################################

import sys


import torch
import torch.nn as nn

sys.path.append("../../../Common")
sys.path.append("../../Config/Src")
from Logging import Logging
from Logging import LogLevel

from Config import Config
from Config import AttentionConfig
from Config import TransformerEncoderConfig
from LayerNormalization import LayerNormalization
from Attention import MultiHeadAttention
from FeedForwardLayer import FeedForwardLayer

class TransformerEncoder(nn.Module):

    __config                        : Config
    __layerNorm1                    : LayerNormalization
    __multiHeadAttention            : MultiHeadAttention
    __layerNorm2                    : LayerNormalization
    __feedForwardLayer              : FeedForwardLayer
    __postMultiHeadAttentionDropout : nn.Dropout
    __postFeedForwardDropoutRate    : nn.Dropout
    __device                        : str
    __transformerEncoderConfig      : TransformerEncoderConfig
    __attentionConfig               : AttentionConfig
    __usePreLayerNorm               : bool = True  # Default to pre-layer normalization


    def __init__(self, config : Config, device: str = "cpu"):
        super().__init__()

        self.__config = config

        self.__transformerEncoderConfig = self.__config.getTransformerEncoderConfig()
        self.__attentionConfig = self.__config.getAttentionConfig()

        self.__usePreLayerNorm = self.__config.usePreNormalisation()
        self.__layerNorm1 = LayerNormalization(config)

        self.__multiHeadAttention =\
            MultiHeadAttention(
                config.getEmbeddingDimension(),
                config.getEmbeddingDimension(),
                config.getContextLength(),
                config.getAttentionHeads(),
                config.useQueryKeyValueBias(),
                self.__attentionConfig.getMultiHeadAttentionDropoutRate())

        self.__layerNorm2 = LayerNormalization(config)
        self.__feedForwardLayer = FeedForwardLayer(config)
        self.__postMultiHeadAttentionDropout =\
            nn.Dropout(
                self.__transformerEncoderConfig.getPostMultiHeadAttentionDropoutRate())

        self.__postFeedForwardDropout =\
            nn.Dropout(
                self.__transformerEncoderConfig.getPostFeedForwardDropoutRate())

        Logging.GetInstance().Debug(f"TransformerEncoder initialized with config: {self.__config.getConfigType()}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        Logging.GetInstance().Debug(f"inputs shape : {inputs.shape}")
        residual1 = inputs
        if self.__usePreLayerNorm:
            # Multi-head attention layer with pre layer normalization
            inputs = self.__layerNorm1(inputs)
        attentionOutput = self.__multiHeadAttention(inputs)
        attentionOutput = self.__postMultiHeadAttentionDropout(attentionOutput)
        attentionOutput = attentionOutput + residual1
        if not self.__usePreLayerNorm:
            # If post layer normalization is used apply normalization here
            attentionOutput = self.__layerNorm1(attentionOutput)
        Logging.GetInstance().Debug(f"outputs after attention : {attentionOutput.shape}")

        residual2 = attentionOutput
        if self.__usePreLayerNorm:
            # Feed forward layer with pre layer normalization
            attentionOutput = self.__layerNorm2(attentionOutput)
        feedForwardOutput = self.__feedForwardLayer(attentionOutput)
        feedForwardOutput = self.__postFeedForwardDropout(feedForwardOutput)
        output = feedForwardOutput + residual2
        if not self.__usePreLayerNorm:
            # If post layer normalization is used apply normalization here
            output = self.__layerNorm2(output)
        Logging.GetInstance().Debug(f"outputs after feed forward : {output.shape}")
        return output

    def freezePostMultiHeadAttentionDropout(self):
        for param in self.__postMultiHeadAttentionDropout.parameters():
            param.requires_grad = False
        Logging.GetInstance().Info("Post Multi Head Attention Dropout layer frozen successfully")

    def freezePostFeedForwardDropout(self):
        for param in self.__postFeedForwardDropout.parameters():
            param.requires_grad = False
        Logging.GetInstance().Info("Post Feed Forward Dropout layer frozen successfully")

    def getMultiHeadAttention(self) -> MultiHeadAttention:
        return self.__multiHeadAttention

    def getLayerNorm1(self) -> LayerNormalization:
        return self.__layerNorm1

    def getLayerNorm2(self) -> LayerNormalization:
        return self.__layerNorm2

    def getFeedForwardLayer(self) -> FeedForwardLayer:
        return self.__feedForwardLayer

