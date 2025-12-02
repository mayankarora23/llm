#######################################################################
# File - TokenEmbedding.py
# Author - Mayank Arora
# Description - This file contains the TokenEmbedding class which is used to create the token embedding
#               using the sliding window technique.
#######################################################################
import sys

sys.path.append("../../Common")
sys.path.append("../Config/Src")

from Logging import Logging
from Logging import LogLevel
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import random
from torch.utils.data import Dataset, DataLoader, DistributedSampler
#import tiktoken
from transformers import AutoTokenizer

from Config import Config

class GptDataSet(Dataset):
    __tokenizer : AutoTokenizer
    __rawText : str
    __tokenIds : List[torch.Tensor]
    __inputIds : List[torch.Tensor]
    __targetIds : List[torch.Tensor]
    __device : str
    def __init__(self, device: str = "cpu"):
        self.__tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        self.__tokenIds = []
        self.__rawText = ""
        self.__inputIds = []
        self.__targetIds = []

        Logging.GetInstance().Debug("GptDataSet initialized with tokenizer: gpt2")

    def getTokenizer(self):
        return self.__tokenizer

    def setRawText(self, rawText):
        self.__rawText = rawText

    def tokenize(self):
        import re
        SPECIAL_TOKENS = {
            "<|endoftext|>": 50256  # GPT-2's known token
            }
        text = self.__rawText

        # Split while preserving special tokens
        pattern = r"(<\|endoftext\|>)"
        segments = re.split(pattern, text)

        token_ids = []
        for segment in segments:
            if segment in SPECIAL_TOKENS:
                token_ids.append(SPECIAL_TOKENS[segment])
            else:
                token_ids.extend(self.__tokenizer.encode(segment))

        device = next(self.parameters()).device
        self.__tokenIds = torch.tensor(token_ids, device=device)
        Logging.GetInstance().Debug(f"Tokenized text: {self.__tokenIds[:5]}")

    def buildInputTargetPair(self, maxLength : int = 256, stride : int = 128):
        for i in range(0, len(self.__tokenIds) - maxLength, stride):
            inputIds = self.__tokenIds[i:i + maxLength]
            targetIds = self.__tokenIds[i + 1:i + maxLength + 1]
            device = next(self.parameters()).device
            self.__inputIds.append(torch.Tensor(inputIds, device=device))
            self.__targetIds.append(torch.Tensor(targetIds, device=device))

    def __len__(self):
        return len(self.__inputIds)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        inputIds = self.__inputIds[index].long()
        targetIds = self.__targetIds[index].long()
        return inputIds, targetIds

class GptDataLoader:
    __dataSet : GptDataSet
    __dataLoader : DataLoader
    __dataIterator = None
    def __init__(self, dataset : GptDataSet,
                 batchSize : int = 32,
                 shuffle : bool = True,
                 dropLast : bool = True,
                 numWorkers : int = 0):
        self.__dataSet = GptDataSet()
        self.__dataLoader = DataLoader(
            dataset,
            batch_size=batchSize,
            shuffle=shuffle,
            drop_last=dropLast,
            num_workers=numWorkers)

    def getDataLoader(self):
        return self.__dataLoader

    def initIterator(self):
        self.__dataIterator = iter(self.__dataLoader)

    def getNextBatch(self):
        if self.__dataIterator is None:
            Logging.GetInstance().Fatal("Data iterator is not initialized")
            return None
        try:
            batch = next(self.__dataIterator)
            Logging.GetInstance().Debug(f"Batch: {batch}")
            return batch
        except StopIteration:
            Logging.GetInstance().Debug("End of data iterator")
        return None

    def getBatchSize(self):
        if self.__dataLoader is None:
            Logging.GetInstance().Fatal("Data loader is not initialized")
            return None
        return self.__dataLoader.batch_size

class EncoderDecoderDataSet(Dataset):
    __tokenizer : AutoTokenizer
    __rawText : str
    __inputIds : List[torch.Tensor]
    __targetIds : List[torch.Tensor]
    __inputTargetIdPairs : List[tuple]
    __totalTokens : int
    __device : str
    __config: Config
    __runTimeDenoising : bool = False  # Flag to indicate if runtime denoising is enabled
    __inputTargetPair : bool = False  # Flag to indicate if input-target pairs are used
    __inputTextList = List[str]
    __fullInputTextList = List[str]
    __maxLength: int = 256
    __startVariableDataIndex: int = 0
    __batchSize: int = 32
    def __init__(self, device: str = "cpu", config: Config = None):
        self.__tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.__tokenIds = []
        self.__rawText = ""
        self.__inputIds = []
        self.__targetIds = []
        self.__inputTargetIdPairs = []
        self.__totalTokens = 0
        self.SPECIAL_TOKENS = {}
        self.__config = config
        self.__runTimeDenoising = False
        self.__inputTargetPair = False
        self.__inputTextList = []
        self.__fullInputTextList = []
        self.__maxLength = 256
        self.__startVariableDataIndex = self.__config.getDataSetConfig().getEndOfFixedData()
        self.__batchSize = self.__config.getLearningConfig().getBatchSize()
        Logging.GetInstance().Debug("GptDataSet initialized with tokenizer")

    def getTokenizer(self):
        return self.__tokenizer

    def setRawText(self, rawText):
        self.__rawText = rawText

    def __applyMask(self, inputText: str, mask: str = "<extra_id_0>", maskRatio: float = 0.3) -> str:
        """
        BART-style masking for T5 tokenizer:
        - Each masked span replaced by exactly one <extra_id_0>
        - No consecutive masks
        - Do not mask trailing punctuation
        - Mask ratio ~ maskRatio
        """
        # Split words while keeping punctuation separate
        import re
        import random
        import string

        tokens = re.findall(r'\w+|[^\w\s]', inputText, re.UNICODE)
        n = len(tokens)
        maskedTokens = []

        i = 0
        maskedCount = 0
        maskBlock = 0  # tokens after a mask where masking is forbidden

        while i < n:
            remaining = n - i
            token = tokens[i]

            if maskBlock > 0:
                maskedTokens.append(token)
                i += 1
                maskBlock -= 1
                continue

            # Only mask alphabetic tokens (skip pure punctuation)
            if token.isalpha() and random.random() < maskRatio and maskedCount < maskRatio * n:
                spanLen = random.randint(3, 6)
                spanLen = min(spanLen, remaining)

                # Count how many alphabetic tokens in span
                alphaTokens = [t for t in tokens[i:i+spanLen] if t.isalpha()]
                if len(alphaTokens) == 0:
                    maskedTokens.append(token)
                    i += 1
                    continue

                # Add single mask token
                maskedTokens.append(mask)
                i += spanLen
                maskedCount += len(alphaTokens)

                # Avoid consecutive masks
                maskBlock = random.randint(1, 2)
            else:
                maskedTokens.append(token)
                i += 1

        # Rejoin tokens with space but avoid space before punctuation
        result = ''
        for t in maskedTokens:
            if t in string.punctuation:
                result += t
            else:
                if result:
                    result += ' '
                result += t
        return result

    def __populateVariableInputTexts(self):
        if self.__config.getDataSetConfig().useVariableDataSet():
            maxIndex = len(self.__fullInputTextList)
            #startIndex = self.__config.getDataSetConfig().getEndOfFixedData()
            #inputTextIndex = startIndex
            inputTextIndex = 0
            variableTextSize = self.__config.getDataSetConfig().getTotalDataList() - self.__config.getDataSetConfig().getEndOfFixedData()
            while inputTextIndex < variableTextSize:
                #index = random.randint(startIndex, maxIndex - 1)
                index = inputTextIndex + self.__startVariableDataIndex
                inputText = self.__fullInputTextList[index]
                self.__inputTextList.append(inputText)
                inputTextIndex += 1
            Logging.GetInstance().Info(f"Added {inputTextIndex} variable input texts, so length of input text list is {len(self.__inputTextList)}")
            Logging.GetInstance().Info(f"Added variable input from index {self.__startVariableDataIndex} to {index}")
            self.__startVariableDataIndex += inputTextIndex

    def __populateFixedInputTexts(self):
        index = self.__config.getDataSetConfig().getStartOfFixedData() - 1
        while index < self.__config.getDataSetConfig().getEndOfFixedData():
            inputText = self.__fullInputTextList[index]
            index += 1
            self.__inputTextList.append(inputText)
        Logging.GetInstance().Info(f"Added {index} fixed input texts")

    def __populateInputTexts(self):
        self.__inputTextList = []
        self.__populateFixedInputTexts()
        self.__populateVariableInputTexts()

    def __fetchFromInputTexts(self, inputTexts):
        self.__runTimeDenoising = True
        for inputText in inputTexts:
            inputText = inputText.get("Text")

            self.__fullInputTextList.append(inputText)

    def __getInputTargetTokens(self, inputText, targetText) -> Tuple[torch.Tensor, torch.Tensor]:
        import re
        inputText = inputText + "<endoftext>"
        pattern = r"(<endoftext>|<sep>|<mask>|<bos>)"
        segments = re.split(pattern, inputText)

        Logging.GetInstance().Debug(f"Input text: {inputText}")
        Logging.GetInstance().Debug(f"Target text: {targetText}")

        # Build up input tokens
        inputTokenIds = []
        countSegments = 0
        for segment in segments:
            if countSegments < len(segments) - 1:
                if segment in self.SPECIAL_TOKENS:
                    inputTokenIds.append(self.SPECIAL_TOKENS[segment])
                else:
                    inputTokenIds.extend(self.__tokenizer.encode(segment))
            countSegments += 1
        inputTokenIds = inputTokenIds[:self.__maxLength]

        if len(inputTokenIds) < self.__maxLength:
            paddingLength = self.__maxLength - len(inputTokenIds)
            inputTokenIds = inputTokenIds + [0] * paddingLength  # Padding with zeros

        inputTokenIds = torch.tensor(inputTokenIds)
        Logging.GetInstance().Debug(f"Input tokenized text: {inputTokenIds[:5]}")

        self.__totalTokens += len(inputTokenIds)

        # Build up target tokens
        # Split while preserving special tokens
        pattern = r"(<endoftext>|<sep>|<mask>|<bos>)"
        segments = re.split(pattern, targetText)

        targetTokenIds = []
        countSegments = 0
        for segment in segments:
            if countSegments < len(segments) - 1:
                if segment in self.SPECIAL_TOKENS:
                    targetTokenIds.append(self.SPECIAL_TOKENS[segment])
                else:
                    targetTokenIds.extend(self.__tokenizer.encode(segment))
            countSegments += 1

        targetTokenIds = targetTokenIds[:self.__maxLength]

        if len(targetTokenIds) < self.__maxLength:
            paddingLength = self.__maxLength - len(targetTokenIds)
            targetTokenIds = targetTokenIds + [0] * paddingLength  # Padding with zeros

        targetTokenIds = torch.tensor(targetTokenIds)
        Logging.GetInstance().Debug(f"Target tokenized text: {targetTokenIds[:5]}")
        return inputTokenIds, targetTokenIds

    def __populateFixedInputTargetPairs(self, pairs):
        startOfFixedData = self.__config.getDataSetConfig().getStartOfFixedData() - 1
        endOfFixedData = self.__config.getDataSetConfig().getEndOfFixedData()
        for index in range(startOfFixedData, endOfFixedData):
            pair = pairs[index]
            inputText = pair.get("Input")
            targetText = pair.get("Target")
            inputTokenIds, targetTokenIds = self.__getInputTargetTokens(inputText, targetText)
            inputTargetPair = (inputTokenIds, targetTokenIds)
            self.__inputTargetIdPairs.append(inputTargetPair)
        Logging.GetInstance().Info(f"Added {endOfFixedData - startOfFixedData} fixed input-target pairs and len {len(self.__inputTargetIdPairs)}")

    def __populateVariableInputTargetPairs(self, pairs):
        endOfFixedData = self.__config.getDataSetConfig().getEndOfFixedData()

        variableTextSize = self.__config.getDataSetConfig().getTotalDataList() -\
                endOfFixedData
        inputTextIndex = 0
        while inputTextIndex < variableTextSize:
            #index = random.randint(startIndex, maxIndex - 1)
            index = inputTextIndex + self.__startVariableDataIndex
            pair = pairs[index]
            inputText = pair.get("Input")
            targetText = pair.get("Target")
            inputTokenIds, targetTokenIds = self.__getInputTargetTokens(inputText, targetText)
            inputTargetPair = (inputTokenIds, targetTokenIds)
            self.__inputTargetIdPairs.append(inputTargetPair)
            inputTextIndex += 1
            if inputTextIndex + self.__startVariableDataIndex >= len(pairs) and\
                inputTextIndex < variableTextSize:
                # Wrap around if we exceed available pairs
                self.__startVariableDataIndex = endOfFixedData - inputTextIndex
        Logging.GetInstance().Info(f"Added {inputTextIndex} variable input texts")
        Logging.GetInstance().Info(f"Added variable input from index {self.__startVariableDataIndex} to {index}")
        Logging.GetInstance().Info(f"Total input-target pairs length is {len(self.__inputTargetIdPairs)}")
        self.__startVariableDataIndex += inputTextIndex

    def __populateInputTargetPairs(self, pairs):
        self.__inputIds = []
        self.__targetIds = []
        self.__populateFixedInputTargetPairs(pairs)
        self.__populateVariableInputTargetPairs(pairs)

    def __fetchFromInputTargetPairs(self, pairs, maxLength):
        import re
        self.__maxLength = maxLength
        self.__inputTargetPair = True
        for pair in pairs:
            self.__fullInputTextList.append(pair)
        self.__populateInputTargetPairs(self.__fullInputTextList)

    def __fetchFromTexts(self, texts, maxLength, stride):
        import re
        for text in texts:
            inputText = text.get("Text")
            pattern = r"(<endoftext>|<sep>|<mask>|<bos>)"
            segments = re.split(pattern, inputText)

            tokenIds = []
            countSegments = 0
            for segment in segments:
                if countSegments < len(segments) - 1:
                    if segment in self.SPECIAL_TOKENS:
                        tokenIds.append(self.SPECIAL_TOKENS[segment])
                    else:
                        tokenIds.extend(self.__tokenizer.encode(segment))
                countSegments += 1

            tokenIds = tokenIds[:maxLength]

            if len(tokenIds) < maxLength:
                paddingLength = maxLength - len(tokenIds)
                tokenIds = tokenIds + [0] * paddingLength  # Padding with zeros

            device = next(self.parameters()).device
            tokenIds = torch.tensor(tokenIds, device=device)
            Logging.GetInstance().Debug(f"Target tokenized text: {tokenIds[:5]}")

            inputTokenIds = torch.Tensor(tokenIds[:-1], device=device)
            targetTokenIds = torch.Tensor(tokenIds[1:], device=device)

            self.__totalTokens += len(tokenIds)
            self.__inputIds.append(inputTokenIds)
            self.__targetIds.append(targetTokenIds)

    def __getInputTargetPairWithDenoising(self, inputText: str) -> Tuple[torch.Tensor, torch.Tensor]:
        targetText = inputText + "<endoftext>"
        inputText = self.__applyMask(inputText,
                                     mask="<extra_id_0>",
                                     maskRatio=self.__config.getLearningConfig().getDenoisingMaskingRate())
        return self.__getInputTargetTokens(inputText, targetText)

    def buildInputTargetPair(self, maxLength: int = 256, stride: int = 128):
        import re
        import json
        vocabSize = 32128
        self.SPECIAL_TOKENS = {
            "<endoftext>": vocabSize + 1,  # GPT-2's known token
            "<bos>": vocabSize + 2,
            "<sep>": vocabSize + 3,
            "<mask>": vocabSize + 4
        }
        text = self.__rawText

        inputTargetPairsJson = json.loads(text)

        pairs = inputTargetPairsJson.get("InputTargetPairs")
        texts = inputTargetPairsJson.get("Texts")
        inputTexts = inputTargetPairsJson.get("InputTexts")
        if pairs:
            self.__fetchFromInputTargetPairs(pairs, maxLength)
        elif texts:
            self.__fetchFromTexts(texts, maxLength, stride)
        elif inputTexts:
            self.__fetchFromInputTexts(inputTexts)
            self.__populateInputTexts()
        else:
            Logging.GetInstance().Fatal("No input-target pairs found.")

    def __len__(self):
        if self.__runTimeDenoising:
            return len(self.__inputTextList)
        elif self.__inputTargetPair:
            return len(self.__inputTargetIdPairs)
        else:
            return len(self.__inputIds)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.__runTimeDenoising:
            inputText = self.__inputTextList[index]
            inputIds, targetIds = self.__getInputTargetPairWithDenoising(inputText)
            inputIds = inputIds.long()
            targetIds = targetIds.long()
        elif self.__inputTargetPair:
            inputIds, targetIds = self.__inputTargetIdPairs[index]
            inputIds = inputIds.long()
            targetIds = targetIds.long()
        else:
            inputIds = self.__inputIds[index]
            targetIds = self.__targetIds[index]
        if index == len(self.__inputIds) - 1:
            if self.__runTimeDenoising:
                Logging.GetInstance().Debug(f"Last item fetched from dataset at index {index}")
                self.__populateInputTexts()
                random.shuffle(self.__inputTextList)
            if not self.__inputTargetPair:
                Logging.GetInstance().Debug(f"Last item fetched from dataset at index {index}")
                self.__populateInputTargetPairs(self.__fullInputTextList)
                random.shuffle(self.__inputTargetIdPairs)
        return inputIds, targetIds

class EncoderDecoderDataLoader:
    __dataLoader: DataLoader
    __dataIterator = None
    __dataSet: EncoderDecoderDataSet

    def __init__(self, dataset: EncoderDecoderDataSet,
                 batchSize: int = 32,
                 shuffle: bool = True,
                 dropLast: bool = True,
                 numWorkers: int = 0,
                 sampler:DistributedSampler = None):
        shuffle = False
        if sampler is not None:
            Logging.GetInstance().Info("Using DistributedSampler for DataLoader hence disabling shuffling")
            shuffle = False  # When using a sampler, disable shuffling
        self.__dataLoader = DataLoader(
            dataset,
            batch_size=batchSize,
            shuffle=shuffle,
            drop_last=dropLast,
            num_workers=numWorkers,
            sampler=sampler)
        self.__dataSet = dataset

    def getDataLoader(self):
        return self.__dataLoader

    def initIterator(self):
        self.__dataIterator = iter(self.__dataLoader)

    def getNextBatch(self):
        if self.__dataIterator is None:
            Logging.GetInstance().Fatal("Data iterator is not initialized")
            return None
        try:
            batch = next(self.__dataIterator)
            Logging.GetInstance().Debug(f"Batch: {batch}")
            return batch
        except StopIteration:
            Logging.GetInstance().Debug("End of data iterator")
        return None

    def getBatchSize(self):
        if self.__dataLoader is None:
            Logging.GetInstance().Fatal("Data loader is not initialized")
            return None
        return self.__dataLoader.batch_size