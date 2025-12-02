#######################################################################
# File - GptImplement.py
# Author - Mayank Arora
# Description - This file contains the implementation of GPT model
#               based on GPT2 architecture
#######################################################################

import sys
from typing import List


import torch
import torch.nn as nn

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

from DecoderStack import DecoderStack

class GptImplement(nn.Module):
    __config: Config
    __dropout: nn.Dropout
    __decoderStack: DecoderStack
    __layerNorm: LayerNormalization
    __outputLayer: nn.Linear
    __dataLoader: GptDataLoader
    __optimizer: torch.optim.Optimizer
    __numEpochs: int = 4
    __device: str

    def __init__(self, configFileName, device: str = "cpu", numEpochs: int = 4):
        super().__init__()

        self.__config = Config(configFileName)
        self.__config.loadConfig()

        self.__device = device
        self.__numEpochs = numEpochs

        self.to(torch.device(self.__device))

        # Initialize dropout
        self.__dropout = nn.Dropout(self.__config.getDropoutRate()).to(torch.device(self.__device))
        Logging.GetInstance().Debug(f"Dropout initialized with rate: {self.__config.getDropoutRate()}")

        self.__decoderStack = DecoderStack(self.__config, self.__device)

        # Initialize layer normalization
        self.__layerNorm = LayerNormalization(self.__config).to(torch.device(self.__device))

        # Initialize output layer
        self.__outputLayer = nn.Linear(
            self.__config.getEmbeddingDimension(),
            self.__config.getVocabSize(),
            bias=False
        ).to(torch.device(self.__device))

        self.__dataLoader = None
        self.__optimizer = torch.optim.AdamW(self.parameters(), lr=0.0004, weight_decay=0.1)

        Logging.GetInstance().Debug(f"GptImplement initialized with config - {self.__config.getConfigType()}")

    def setDataLoader(self, dataLoader: GptDataLoader):
        Logging.GetInstance().Debug(f"Setting data loader: {dataLoader}")
        self.__dataLoader = dataLoader

    def forward(self, inputIds: torch.Tensor) -> torch.Tensor:
        batchSize, contextLength = inputIds.shape
        Logging.GetInstance().Debug(f"InputIds Shape : {inputIds.shape}")
        decoderOutput = self.__decoderStack(inputIds)
        normalizedDecoderOutput = self.__layerNorm(decoderOutput)
        logits = self.__outputLayer(normalizedDecoderOutput)
        Logging.GetInstance().Debug(f"Logits shape: {logits.shape}")
        return logits

    def __softmaxWithTemperature(self, logits: torch.Tensor, temperature: float = 1.5, dim=-1, topK=3) -> torch.Tensor:

        if topK > 0:
            # If topK is specified, we only keep the topK logits and set the rest to a very low value
            topKValues, _ = torch.topk(logits, topK, dim=-1)
            minTopKValue = torch.min(topKValues, dim=-1, keepdim=True).values
            logits[logits < minTopKValue] = float('-inf')

        if temperature > 0.0:
            # This function applies softmax with temperature scaling to the logits.
            # Temperature scaling is a technique used to control the randomness of predictions.
            # A higher temperature (>1) makes the distribution more uniform, while a lower temperature (<1) makes it sharper
            scaledLogits = logits / temperature

        probabilityVector = torch.softmax(scaledLogits, dim=-1)
        Logging.GetInstance().Debug(f"Probability vector shape: {probabilityVector.shape}")

        return probabilityVector

    def __getNextToken(self, logits: torch.Tensor) -> int:
        Logging.GetInstance().Debug(f"Input IDs shape: {logits.shape}")

        probabilityVector = logits[:, - 1, :]

        probabilityVector = self.__softmaxWithTemperature(probabilityVector, temperature=1, dim=-1)
        #predictedToken = torch.argmax(probabilityVector, dim=-1, keepdim=True)
        predictedToken = torch.multinomial(probabilityVector, num_samples=1)
        Logging.GetInstance().Debug(f"Predicted token: {predictedToken}")

        return predictedToken

    def getLoss(self, logits: torch.Tensor, targetIds: torch.Tensor) -> torch.Tensor:
        Logging.GetInstance().Debug(f"Logit shape: {logits.shape}")

        logitsFlat = logits.flatten(start_dim=0, end_dim=1)
        targetIdsFlat = targetIds.flatten(start_dim=0, end_dim=1)
        Logging.GetInstance().Debug(f"Logits flat shape: {logitsFlat.shape}, Target IDs flat shape: {targetIdsFlat.shape}")

        loss = torch.nn.functional.cross_entropy(logitsFlat, targetIdsFlat)
        Logging.GetInstance().Debug(f"Loss: {loss.item()}")

        return loss

    def getPerplexity(self, loss) -> float:
        Logging.GetInstance().Debug(f"Calculating perplexity from loss: {loss.item()}")
        perplexity = torch.exp(loss)
        Logging.GetInstance().Debug(f"Perplexity: {perplexity}")
        return perplexity

    def loadParameters(self, loadFileName: str):
        loadFileHandler = open(loadFileName, "rb")
        missing_keys, unexpected_keys = self.load_state_dict(torch.load(loadFileHandler), strict=False)
        Logging.GetInstance().Info(f"Missing keys in loaded model: {missing_keys}")
        Logging.GetInstance().Info(f"Unexpected keys in loaded model: {unexpected_keys}")
        Logging.GetInstance().Info(f"Model parameters loaded from {loadFileName}")
        loadFileHandler.close()

    def __getTokenConfidence(self, logits: torch.Tensor, targetIds: torch.Tensor) -> float:
        # Calculating and printing token confidence -
        probs = torch.nn.functional.softmax(logits, dim=-1)
        tokenConfidence = probs.gather(dim=2, index=targetIds.unsqueeze(-1)).squeeze(-1)
        avgConf = tokenConfidence.mean().item()

        return avgConf

    def runTraining(
            self,
            saveFileName: str = "",
            loadFileName: str = "",
            numEpochs: int = 4,
            isCheckpoint: bool = False):
        loadFileHandler = None
        if loadFileName:
            loadFileHandler = open(loadFileName, "rb")
            self.load_state_dict(torch.load(loadFileHandler))
            Logging.GetInstance().Info(f"Model loaded from {loadFileName}")

        for epoch in range(numEpochs):
            count = 0
            dataLoadLength = len(self.__dataLoader.getDataLoader())
            for inputIds, targetIds in self.__dataLoader.getDataLoader():
                self.__optimizer.zero_grad()
                logits = self.forward(inputIds)
                Logging.GetInstance().Debug(f"Logits shape: {logits.shape}")

                loss = self.getLoss(logits, targetIds)
                Logging.GetInstance().Debug(f"Loss: {loss.item()}")

                perplexity = self.getPerplexity(loss)

                loss.backward()
                self.__optimizer.step()

                learningRateUsed = self.__optimizer.param_groups[0]['lr']
                avgConf = self.__getTokenConfidence(logits, targetIds)

                if count % 100 == 0 and isCheckpoint and saveFileName:
                    torch.save(self.state_dict(), saveFileName)
                    Logging.GetInstance().Info(f"Model checkpoint saved to {saveFileName}")

                Logging.GetInstance().Info(
                    f"{count}/{dataLoadLength} [{epoch + 1}/{numEpochs}] "
                    f"Loss: {loss} Perplexity: {perplexity:.2f}, "
                    f"LR: {learningRateUsed:.6f}, tokenConfidence: {avgConf:.4f}")
                count += 1


            if isCheckpoint and saveFileName:
                torch.save(self.state_dict(), saveFileName)
                Logging.GetInstance().Info(f"Model checkpoint saved to {saveFileName}")
            Logging.GetInstance().Info(f"Epoch {epoch + 1}/{self.__numEpochs} completed")

        if not isCheckpoint and saveFileName:
            torch.save(self.state_dict(), saveFileName)


    def generateText(self, startingInput: torch.Tensor, maxNewToken: int) -> torch.Tensor:
        Logging.GetInstance().Debug(f"Starting input shape: {startingInput.shape}")
        input = startingInput.clone()

        for _ in range(maxNewToken):
            inputTruncated = input[:, -self.__config.getContextLength():]
            Logging.GetInstance().Debug(f"Input truncated shape: {inputTruncated.shape}")
            logits = self.forward(inputTruncated)
            Logging.GetInstance().Debug(f"Logits shape: {logits.shape}")
            nextToken = self.__getNextToken(logits)
            Logging.GetInstance().Debug(f"Next token: {nextToken}")
            input = torch.cat((input, nextToken), dim=1)

            # Break if any sentence reaches an end token
            if torch.any(nextToken == 50256):
                Logging.GetInstance().Debug("Special token encountered. Stopping generation.")
                break


        Logging.GetInstance().Debug(f"Final generated input shape: {input.shape}")
        return input
