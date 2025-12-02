#######################################################################
# File - DecoderStack.py
# Author - Mayank Arora
# Description - This file contains the implementation of transformer
#               decoder stack using multi head attention.
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
from TransformerDecoder import TransformerDecoder
from LayerNormalization import LayerNormalization

from TokenEmbedding import GptDataSet
from TokenEmbedding import GptDataLoader
from LayerNormalization import LayerNormalization

class DecoderStack(nn.Module):
    __config: Config
    __decoderTokenEmbedding: nn.Embedding
    __decoderPositionEmbedding: nn.Embedding
    __dropout: nn.Dropout
    __transformerDecoderBlocks: List[TransformerDecoder]
    __device: str
    __layerNormDecoder : LayerNormalization
    __outputNorm : LayerNormalization
    __isFrozen: bool = False  # Flag to check if the decoder stack is frozen
    __decoderLayerDropRate: float = 0.0  # Default layer drop rate for decoder

    def __initDropoutAndNorm(self):
        self.__layerNormDecoder = LayerNormalization(self.__config)

        Logging.GetInstance().Debug(f"Position embedding initialized with context length: "
                                    f"{self.__config.getContextLength()} and embedding dimension: "
                                    f"{self.__config.getEmbeddingDimension()}")

        self.__dropout = nn.Dropout(self.__config.getDropoutRate())
        Logging.GetInstance().Debug(f"Dropout initialized with rate: {self.__config.getDropoutRate()}")

        self.__decoderLayerDropRate = self.__config.getLearningConfig().getDecoderLayerDropRate()
        Logging.GetInstance().Info(f"Decoder layer drop rate set to: {self.__decoderLayerDropRate}")

    def __initTransformerDecoder(self):
        # Initialize transformer blocks
        self.__transformerDecoderBlocks = nn.ModuleList(
            [
                TransformerDecoder(self.__config) for _ in range(self.__config.getNumLayers())
            ]
        )
        self.__outputNorm = LayerNormalization(self.__config)

    def __initTokenEmbeddings(self):
        self.__decoderTokenEmbedding = nn.Embedding(
            self.__config.getVocabSize(),
            self.__config.getEmbeddingDimension()
        )
        Logging.GetInstance().Debug(f"Token embedding initialized with vocab size: {self.__config.getVocabSize()} "
                                    f"and embedding dimension: {self.__config.getEmbeddingDimension()}")
        self.__decoderPositionEmbedding = nn.Embedding(
            self.__config.getContextLength(),
            self.__config.getEmbeddingDimension()
        )
        Logging.GetInstance().Debug(f"Position embedding initialized with context length: "
                                    f"{self.__config.getContextLength()} and embedding dimension: "
                                    f"{self.__config.getEmbeddingDimension()}")

    def __init__(self, config: Config, device: str = "cpu"):
        super().__init__()

        self.__config = config

        self.__initTokenEmbeddings()
        self.__initDropoutAndNorm()
        self.__initTransformerDecoder()

        Logging.GetInstance().Debug(f"DecoderStack initialized with config - {self.__config.getConfigType()}")


    def __getTokenEmbedding(self, inputs: torch.Tensor) -> torch.Tensor:

        # Token and position embeddings for decoder
        batchSize, contextLength = inputs.shape
        device = next(self.parameters()).device
        self.device = device
        decoderTokenEmbeddings = self.__decoderTokenEmbedding(inputs).to(device)
        decoderPositionEmbeddings = self.__decoderPositionEmbedding(
            torch.arange(contextLength, device=device))
        decoderEmbeddings = decoderTokenEmbeddings + decoderPositionEmbeddings

        return decoderEmbeddings

    def forward(self, inputs: torch.Tensor, encoderOutputs: torch.Tensor = None) -> torch.Tensor:
        # Here it is assumed that inputs received are layer normalized after token and position embedding.
        Logging.GetInstance().Debug(f"inputs shape : {inputs.shape}")

        # Get token embeddings
        decoderEmbeddings = self.__getTokenEmbedding(inputs)

        # Apply layer normalization
        decoderEmbeddings = self.__layerNormDecoder(decoderEmbeddings)
        # Apply dropout
        decoderEmbeddings = self.__dropout(decoderEmbeddings)

        # Pass through transformer decoder blocks
        for transformerDecoder in self.__transformerDecoderBlocks:
            dropChance = random.uniform(0, 1)
            if self.training and dropChance < self.__decoderLayerDropRate:
                Logging.GetInstance().Debug("Skipping a decoder layer due to layer drop.")
                continue
            decoderEmbeddings = transformerDecoder(decoderEmbeddings, encoderOutputs)

        output = decoderEmbeddings

        # Apply output layer normalization
        output = self.__outputNorm(output)

        return output

    def getEntropy(self, epoch, count) -> torch.Tensor:
        # Logging decoder entropy
        avgEntropy = 0.0
        perLayerEntropy = []
        for transformerDecoderBlock in self.__transformerDecoderBlocks:
            attentionWeights = transformerDecoderBlock.getCrossAttentionWeights()
            weights = attentionWeights.clamp(min=1e-8)              # avoid log(0)
            tokenEntropy = -(weights * weights.log()).sum(-1)      # [B, heads, target_len]
            perLayerEntropy.append(tokenEntropy.mean(dim=1))
            avgEntropy = tokenEntropy.mean().item()               # scalar

        entropyLog = f"{epoch}.{count},"

        for entropy in perLayerEntropy:
            entropyLog = entropyLog+f"{entropy.mean().item():.4f},"

        Logging.GetInstance().SetDetailedLogNeeded(False)
        Logging.GetInstance().Info(f"{entropyLog}",logFileName="DecoderEntropy.csv",printToConsole=False)
        Logging.GetInstance().SetDetailedLogNeeded(True)

        return avgEntropy
    
    def setDropoutScaling(self, dropoutScaling: float):
        for transformerDecoder in self.__transformerDecoderBlocks:
            transformerDecoder.setDropoutScaling(dropoutScaling)
        device = next(self.parameters()).device
        self.__dropout = nn.Dropout(self.__config.getDropoutRate() * dropoutScaling).to(device)
        Logging.GetInstance().Debug(f"DecoderStack dropout scaling set to: {dropoutScaling}")

    def getTransformerDecoderBlocks(self) -> List[TransformerDecoder]:
        return self.__transformerDecoderBlocks

    def freeze(self):
        Logging.GetInstance().Debug("Freezing DecoderStack parameters.")
        for param in self.parameters():
            param.requires_grad = False
        self.__isFrozen = True

    def unfreeze(self):
        Logging.GetInstance().Debug("Unfreezing DecoderStack parameters.")
        for param in self.parameters():
            param.requires_grad = True
        self.__isFrozen = False

    def freezeDecoderEmbeddings(self):
        Logging.GetInstance().Debug("Freezing DecoderStack token and position embeddings.")
        for param in self.__decoderTokenEmbedding.parameters():
            param.requires_grad = False
        for param in self.__decoderPositionEmbedding.parameters():
            param.requires_grad = False

    def unfreezeDecoderEmbeddings(self):
        Logging.GetInstance().Debug("Unfreezing DecoderStack token and position embeddings.")
        for param in self.__decoderTokenEmbedding.parameters():
            param.requires_grad = True
        for param in self.__decoderPositionEmbedding.parameters():
            param.requires_grad = True

    def isFrozen(self) -> bool:
        return self.__isFrozen
