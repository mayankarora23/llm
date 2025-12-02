#######################################################################
# File - EncoderStack.py
# Author - Mayank Arora
# Description - This file contains the implementation of transformer
#               encoder stack using multi head attention.
#######################################################################

import sys
from typing import List, Tuple


import torch
import torch.nn as nn
import random

sys.path.append("../../../Common")
sys.path.append("../../Config/Src")
sys.path.append("../../Transformers/Src")
sys.path.append("../../DataPreperation")

from Logging import Logging
from Logging import LogLevel

from Config import Config
from TransformerEncoder import TransformerEncoder
from LayerNormalization import LayerNormalization

from TokenEmbedding import GptDataSet
from TokenEmbedding import GptDataLoader
from LayerNormalization import LayerNormalization

class EncoderStack(nn.Module):
    __config: Config
    __encoderTokenEmbedding: nn.Embedding
    __encoderPositionEmbedding: nn.Embedding
    __dropout: nn.Dropout
    __transformerEncoderBlocks: List[TransformerEncoder]
    __device: str
    __layerNormEncoder : LayerNormalization
    __outputNorm : LayerNormalization
    __isFrozen: bool = False  # Flag to check if the encoder stack is frozen
    __encoderLayerDropRate: float = 0.0  # Default layer drop rate for encoder

    def __initDropoutAndNorm(self):
        self.__layerNormEncoder = LayerNormalization(self.__config)

        Logging.GetInstance().Debug(f"Position embedding initialized with context length: "
                                    f"{self.__config.getContextLength()} and embedding dimension: "
                                    f"{self.__config.getEmbeddingDimension()}")
        self.__dropout = nn.Dropout(self.__config.getDropoutRate())
        Logging.GetInstance().Debug(f"Dropout initialized with rate: {self.__config.getDropoutRate()}")

        self.__encoderLayerDropRate = self.__config.getLearningConfig().getEncoderLayerDropRate()
        Logging.GetInstance().Info(f"Encoder layer drop rate set to: {self.__encoderLayerDropRate}")

    def __initTransformerEncoder(self):
        self.__transformerEncoderBlocks = nn.Sequential(
            *[
                TransformerEncoder(self.__config) for _ in range(self.__config.getNumLayers())
            ]
        )
        self.__outputNorm = LayerNormalization(self.__config)

    def __initTokenEmbeddings(self):
        self.__encoderTokenEmbedding = nn.Embedding(
            self.__config.getVocabSize(),
            self.__config.getEmbeddingDimension()
        )

        self.__encoderPositionEmbedding = nn.Embedding(
            self.__config.getContextLength(),
            self.__config.getEmbeddingDimension()
        )

        Logging.GetInstance().Debug(f"Token embedding initialized with vocab size: "
                                    f"{self.__config.getVocabSize()} and embedding dimension: "
                                    f"{self.__config.getEmbeddingDimension()}")
        Logging.GetInstance().Debug(f"Position embedding initialized with context length: "
                                    f"{self.__config.getContextLength()} and embedding dimension: "
                                    f"{self.__config.getEmbeddingDimension()}")

    def __init__(self, config: Config, device: str = "cpu"):
        super().__init__()

        self.__config = config

        self.__initTokenEmbeddings()
        self.__initDropoutAndNorm()
        self.__initTransformerEncoder()

        Logging.GetInstance().Debug(f"EncoderStack initialized with config - {self.__config.getConfigType()}")

    def __getTokenEmbeddingsWithLayerNorm(self, inputs: torch.Tensor) -> torch.Tensor:
        batchSize, contextLength = inputs.shape
        Logging.GetInstance().Debug(f"InputIds Shape : {inputs.shape}")
        # Below code is to generate embeddings for the encoder input

        encoderTokenEmbeddings = self.__encoderTokenEmbedding(inputs)
        encoderPositionEmbeddings = self.__encoderPositionEmbedding(
            torch.arange(contextLength, device=inputs.device)
        )
        encoderEmbeddings = encoderTokenEmbeddings + encoderPositionEmbeddings

        Logging.GetInstance().Debug(f"Encoder embeddings shape: {encoderEmbeddings.shape}")
        return encoderEmbeddings

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoderEmbeddings = self.__getTokenEmbeddingsWithLayerNorm(inputs)
        # Apply layer normalization to the embeddings
        encoderEmbeddings = self.__layerNormEncoder(encoderEmbeddings)
        # Apply dropout to the embeddings
        encoderEmbeddings = self.__dropout(encoderEmbeddings)
        # Pass the embeddings through the transformer encoder blocks

        encoderOutput = encoderEmbeddings
        for encoderBlock in self.__transformerEncoderBlocks:
            dropChance = random.uniform(0, 1)
            # If in training mode and drop chance is less than layer drop rate, skip this layer
            # this is a form of stochastic depth regularization and this helps in training deeper networks
            # this also helps in reducing overfitting and improves generalization.
            if self.training and dropChance < self.__encoderLayerDropRate:
                Logging.GetInstance().Debug("Skipping encoder layer due to layer drop or frozen state.")
                continue
            encoderOutput = encoderBlock(encoderOutput)

        # Apply output layer normalization
        encoderOutput = self.__outputNorm(encoderOutput)

        logits = encoderOutput

        Logging.GetInstance().Debug(f"Encoder output shape: {logits.shape}")

        return logits

    def getTransformerEncoderBlocks(self) -> List[TransformerEncoder]:
        return self.__transformerEncoderBlocks

    def freeze(self):
        Logging.GetInstance().Debug("Freezing EncoderStack parameters.")
        for param in self.parameters():
            param.requires_grad = False
        self.__isFrozen = True

    def unfreeze(self):
        Logging.GetInstance().Debug("Unfreezing EncoderStack parameters.")
        for param in self.parameters():
            param.requires_grad = True
        self.__isFrozen = False

    def freezeEncoderEmbeddings(self):
        Logging.GetInstance().Debug("Freezing EncoderStack token and position embeddings.")
        for param in self.__encoderTokenEmbedding.parameters():
            param.requires_grad = False
        for param in self.__encoderPositionEmbedding.parameters():
            param.requires_grad = False

    def unfreezeEncoderEmbeddings(self):
        Logging.GetInstance().Debug("Unfreezing EncoderStack token and position embeddings.")
        for param in self.__encoderTokenEmbedding.parameters():
            param.requires_grad = True
        for param in self.__encoderPositionEmbedding.parameters():
            param.requires_grad = True

    def isFrozen(self) -> bool:
        return self.__isFrozen