import unittest
import sys
from typing import List

sys.path.append("../../../Common")
sys.path.append("../Src")
sys.path.append("../../Config/Src")

from Logging import Logging
from Logging import LogLevel
import numpy.testing as npt

import torch
import torch.nn as nn
from Config import Config

class DummyTransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dummy forward method for the transformer block
        return x

class DummyLayerNormalization(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dummy forward method for layer normalization
        return x

class DummyGptModel(nn.Module):
    __config: Config
    __tokenEmbedding: nn.Embedding
    __positionEmbedding: nn.Embedding
    __dropout: nn.Dropout
    __transformerBlocks: List[DummyTransformerBlock]
    __layerNorm: DummyLayerNormalization
    __outputLayer: nn.Linear

    def __init__(self, configFileName):
        super().__init__()

        self.__config = Config(configFileName)
        self.__config.loadConfig()

        # Initialize token embeddings and position embeddings
        self.__tokenEmbedding = nn.Embedding(
            self.__config.getVocabSize(),
            self.__config.getEmbeddingDimension()
        )
        self.__positionEmbedding = nn.Embedding(
            self.__config.getContextLength(),
            self.__config.getEmbeddingDimension()
        )
        self.__dropout = nn.Dropout(self.__config.getDropoutRate())

        # Initialize transformer blocks
        self.__transformerBlocks = nn.Sequential(
            *[
                DummyTransformerBlock(self.__config) for _ in range(self.__config.getNumLayers())
            ]
        )

        # Initialize layer normalization
        self.__layerNorm = DummyLayerNormalization(self.__config)

        # Initialize output layer
        self.__outputLayer = nn.Linear(
            self.__config.getEmbeddingDimension(),
            self.__config.getVocabSize(),
            bias=False
        )

        Logging.GetInstance().Debug(f"DummyGptModel initialized with config - {self.__config.getConfigType()}")

    def forward(self, inputIds: torch.Tensor) -> torch.Tensor:
        batchSize, contextLength = inputIds.shape
        Logging.GetInstance().Debug(f"Batch size: {batchSize}, Context length: {contextLength}")
        tokenEmbeddings = self.__tokenEmbedding(inputIds)
        positionEmbeddings = self.__positionEmbedding(
            torch.arange(contextLength, device=inputIds.device)
        )
        embeddings = tokenEmbeddings + positionEmbeddings
        embeddings = self.__dropout(embeddings)
        embeddings = self.__transformerBlocks(embeddings)
        embeddings = self.__layerNorm(embeddings)
        output = self.__outputLayer(embeddings)
        Logging.GetInstance().Debug(f"Output shape: {output.shape}")
        return output


class TestGptModel(unittest.TestCase):

    def testDummyGptModel(self):
        import tiktoken

        tokenizer = tiktoken.get_encoding("gpt2")
        inputText1 = "Every effort moves you"
        inputText2 = "Every day holds a"

        batch = []
        batch.append(torch.tensor(tokenizer.encode(inputText1)))
        batch.append(torch.tensor(tokenizer.encode(inputText2)))

        batch = torch.stack(batch, dim=0)

        torch.manual_seed(123)
        configFileName = "../../Config/Test/TestConfig.json"
        model = DummyGptModel(configFileName)
        output = model(batch)
        Logging.GetInstance().Info(f"Output shape: {output.shape}")
        Logging.GetInstance().Info(f"Output: {output}")
        npt.assert_equal(output.shape, (2, 4, 50257))  # Assuming vocab size is 50257 for GPT-2

def main():
    Logging.GetInstance().SetLogLevel(LogLevel.DEBUG)
    unittest.main()

if __name__ == "__main__":
    main()