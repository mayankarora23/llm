#######################################################################
# File - SelfAttentionImp.py
# Author - Mayank Arora
# Description - This file contains the implementation of self attention
#               mechanism using PyTorch.
#######################################################################

import sys


import torch
import torch.nn as nn

sys.path.append("../../../Common")
from Logging import Logging
from Logging import LogLevel

#######################################################################
# Class - CausalAttentionMask
# Description - This class implements the Causal Attention Masked
#               mechanism
#######################################################################
class CausalAttentionMask(nn.Module):
    wQuery : torch.Tensor
    wKey : torch.Tensor
    wValue : torch.Tensor
    dropout : torch.Tensor
    __device : str

    def __init__(self,
                 dIn : int,
                 dOut : int,
                 contextLength : int,
                 dropout : int,
                 qkvBias : bool = False,
                 device: str = "cpu"):
        super().__init__()

        self.wQuery = nn.Linear(dIn, dOut, bias=qkvBias)
        self.wKey = nn.Linear(dIn, dOut, bias=qkvBias)
        self.wValue = nn.Linear(dIn, dOut, bias=qkvBias)

        Logging.GetInstance().Debug(f"wQuery : {self.wQuery}")
        Logging.GetInstance().Debug(f"wKey : {self.wKey}")
        Logging.GetInstance().Debug(f"wValue : {self.wValue}")

        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(torch.ones(contextLength, contextLength), diagonal=1))

        Logging.GetInstance().Debug(f"SelfAttention initialized with dIn: {dIn}, dOut: {dOut}")

    def forward(self, inputs):
        Logging.GetInstance().Debug(f"inputs shape : {inputs.shape}")
        Logging.GetInstance().Debug(f"inputs :\n{inputs}")
        batchSize, numTokens, dIn = inputs.shape
        keys : torch.Tensor = self.wKey(inputs)
        queries : torch.Tensor = self.wQuery(inputs)
        values : torch.Tensor = self.wValue(inputs)

        Logging.GetInstance().Debug(f"keys :\n{keys}")
        Logging.GetInstance().Debug(f"keys.transpose(1, 2) :\n{keys.transpose(1, 2)}")
        Logging.GetInstance().Debug(f"queries : {queries}")
        Logging.GetInstance().Debug(f"values : {values}")

        # Attention is calculated using dot product which in a way gives the similarity between
        # querry and keys (which in a way are other elements in the sequence)
        attentionScores = queries @ keys.transpose(1, 2)
        attentionScores = attentionScores.masked_fill_(
            self.mask.bool()[:numTokens, :numTokens], -torch.inf)

        Logging.GetInstance().Debug(f"attentionScores :\n{attentionScores}")

        # Now attentionScores is normalized to get attentionWeights, by normalization we are
        # trying to calculate the weights such that sum of all the weights for the query is 
        # 1. We do this so that the results are more interpretable, something like we know
        # in terms of percentage for each token's likelyhood to come as next token to the
        # query We are using softmax for that

        # Also we are not using naive softmax implementation i.e. -
        # softmax = exp(x) / sum(exp(x))
        # that is because this can cause overflow for large values of x
        # So we are using the below implementation which is numerically stable
        # softmax = exp(x - max(x)) / sum(exp(x - max(x))). This already part of
        # torch.softmax implementation. So we are using that.
        maskedAttentionWeights = torch.softmax(attentionScores / keys.shape[-1] ** 0.5, dim=-1)

        maskedAttentionWeights = self.dropout(maskedAttentionWeights)

        Logging.GetInstance().Debug(f"maskedAttentionWeights =\n{maskedAttentionWeights}")
        contextVector = maskedAttentionWeights @ values
        Logging.GetInstance().Debug(f"contextVector =\n{contextVector}")
        return contextVector

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self,
                 dIn : int,
                 dOut : int,
                 contextLength : int,
                 numHeads : int,
                 qkvBias : bool = False,
                 dropout : float = 0.0,
                 device: str = "cpu"):
        super().__init__()

        self.__heads = nn.ModuleList([
            CausalAttentionMask(dIn, dOut, contextLength, dropout, qkvBias)
            for _ in range(numHeads)
        ])

    def forward(self, inputs):
        Logging.GetInstance().Debug(f"inputs shape : {inputs.shape}")
        Logging.GetInstance().Debug(f"inputs :\n{inputs}")
        batchSize, numTokens, dIn = inputs.shape
        Logging.GetInstance().Debug(f"batchSize = {batchSize}, numTokens = {numTokens}, dIn = {dIn}")

        # We are using torch.stack to stack all the heads together
        # and then we are using torch.cat to concatenate the heads
        # together
        contextVector = torch.cat([head(inputs) for head in self.__heads], dim=-1)

        Logging.GetInstance().Debug(f"contextVector shape : {contextVector.shape}")
        Logging.GetInstance().Debug(f"contextVector :\n{contextVector}")
        return contextVector

class MultiHeadAttention(nn.Module):
    __dIn : int
    __dOut : int
    __contextLength : int
    __numHeads : int
    __wQuery : nn.Linear
    __wKey : nn.Linear
    __wValue : nn.Linear
    __wOut : nn.Linear
    __device : str

    def __init__(self,
                 dIn : int,
                 dOut : int,
                 contextLength : int,
                 numHeads : int,
                 qkvBias : bool = False,
                 dropout : float = 0.0,
                 device: str = "cpu"):
        super().__init__()

        self.__dIn = dIn
        self.__dOut = dOut
        self.__contextLength = contextLength
        self.__numHeads = numHeads
        self.__wQuery = nn.Linear(dIn, dOut, bias=qkvBias)
        self.__wKey = nn.Linear(dIn, dOut, bias=qkvBias)
        self.__wValue = nn.Linear(dIn, dOut, bias=qkvBias)
        self.__wOut = nn.Linear(dOut, dOut, bias=qkvBias)

        Logging.GetInstance().Debug(f"wQuery : {self.__wQuery}")
        Logging.GetInstance().Debug(f"wKey : {self.__wKey}")
        Logging.GetInstance().Debug(f"wValue : {self.__wValue}")

        self.__dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(torch.ones(self.__contextLength, self.__contextLength), diagonal=1))

        Logging.GetInstance().Debug(f"SelfAttention initialized with dIn: {dIn}, dOut: {dOut}")

    def forward(self, inputs : torch.Tensor):
        # Inputs received here is in the shape of (batchSize, numTokens, dIn)
        batchSize, numTokens, dIn = inputs.shape
        headDim = self.__dOut // self.__numHeads
        keys = self.__wKey(inputs)
        queries : torch.tensor = self.__wQuery(inputs)
        values : torch.tensor = self.__wValue(inputs)

        # In below code we are changing the shape of the keys, queries and values
        # to (batchSize, numTokens, numHeads, headDim)
        # so we have to take care that dIn is divisible by numHeads
        assert dIn % self.__numHeads == 0, f"dIn {dIn} is not divisible by numHeads {self.__numHeads}"
        keys = keys.view(batchSize, numTokens, self.__numHeads, headDim)
        queries = queries.view(batchSize, numTokens, self.__numHeads, headDim)
        values = values.view(batchSize, numTokens, self.__numHeads, headDim)

        Logging.GetInstance().Debug(f"keys shape : {keys.shape}")
        # Now we are transposing the keys, queries and values from
        # (batchSize, numTokens, numHeads, headDim) to (batchSize, numHeads, numTokens, headDim)
        # basically we are grouping the keys, queries and values by heads, so that we can run the
        # attention mechanism in parallel for all the heads.
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        Logging.GetInstance().Debug(f"keys shape : {keys.shape}")

        # Attention is calculated using dot product which in a way gives the similarity between
        # querry and keys (which in a way are other elements in the sequence). You may notice that
        # transpose is taken for 2, 3 since we are keeping the batch size and numHeads as is
        # The shape of keys is (batchSize, numHeads, numTokens, headDim)
        # The shape of queries is (batchSize, numHeads, numTokens, headDim)
        # The shape of keysTranspose is (batchSize, numHeads, headDim, numTokens)
        keysTranspose = keys.transpose(2, 3)
        Logging.GetInstance().Debug(f"keyTranspose shape : {keysTranspose.shape}")
        attentionScores = queries @ keysTranspose

        Logging.GetInstance().Debug(f"attentionScores :\n{attentionScores}")

        attentionWeights = torch.softmax(attentionScores / headDim ** 0.5, dim=-1)
        attentionWeights = self.__dropout(attentionWeights)
        Logging.GetInstance().Debug(f"attentionWeights shape :\n{attentionWeights.shape}")

        # Now we are doing the dot product between attentionWeights and values
        # to get the context vector
        # The shape of attentionWeights is (batchSize, numHeads, numTokens, numToken)
        # The shape of values is (batchSize, numHeads, numTokens, headDim)
        # The shape of contextVector is (batchSize, numHeads, headDim, numTokens)
        Logging.GetInstance().Debug(f"values shape : {values.shape}")

        # Now the grouping again has changed w.r.t number of tokens
        contextVector = (attentionWeights @ values).transpose(1, 2)
        Logging.GetInstance().Debug(f"contextVector : {contextVector}")
        Logging.GetInstance().Debug(f"contextVector contingous : {contextVector.contiguous()}")

        contextVector = contextVector.contiguous().view(batchSize, numTokens, self.__dOut)

        contextVector = self.__wOut(contextVector)

        Logging.GetInstance().Debug(f"contextVector : {contextVector}")
        return contextVector

    def freezeQuery(self):
        for param in self.__wQuery.parameters():
            param.requires_grad = False

    def unfreezeQuery(self):
        for param in self.__wQuery.parameters():
            param.requires_grad = True

    def freezeKey(self):
        for param in self.__wKey.parameters():
            param.requires_grad = False

    def unfreezeKey(self):
        for param in self.__wKey.parameters():
            param.requires_grad = True

    def freezeValue(self):
        for param in self.__wValue.parameters():
            param.requires_grad = False

    def unfreezeValue(self):
        for param in self.__wValue.parameters():
            param.requires_grad = True

class MaskedMultiHeadAttention(nn.Module):
    __dIn : int
    __dOut : int
    __contextLength : int
    __numHeads : int
    __wQuery : nn.Linear
    __wKey : nn.Linear
    __wValue : nn.Linear
    __wOut : nn.Linear
    __device : str

    def __init__(self,
                 dIn : int,
                 dOut : int,
                 contextLength : int,
                 numHeads : int,
                 qkvBias : bool = False,
                 dropout : float = 0.0,
                 device: str = "cpu"):
        super().__init__()

        self.__dIn = dIn
        self.__dOut = dOut
        self.__contextLength = contextLength
        self.__numHeads = numHeads
        self.__wQuery = nn.Linear(dIn, dOut, bias=qkvBias)
        self.__wKey = nn.Linear(dIn, dOut, bias=qkvBias)
        self.__wValue = nn.Linear(dIn, dOut, bias=qkvBias)
        self.__wOut = nn.Linear(dOut, dOut, bias=qkvBias)

        Logging.GetInstance().Debug(f"wQuery : {self.__wQuery}")
        Logging.GetInstance().Debug(f"wKey : {self.__wKey}")
        Logging.GetInstance().Debug(f"wValue : {self.__wValue}")

        self.__dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(torch.ones(self.__contextLength, self.__contextLength), diagonal=1))

        Logging.GetInstance().Debug(f"SelfAttention initialized with dIn: {dIn}, dOut: {dOut}")

    def forward(self, inputs : torch.Tensor):
        # Inputs received here is in the shape of (batchSize, numTokens, dIn)
        batchSize, numTokens, dIn = inputs.shape
        headDim = self.__dOut // self.__numHeads
        keys = self.__wKey(inputs)
        queries : torch.tensor = self.__wQuery(inputs)
        values : torch.tensor = self.__wValue(inputs)

        # In below code we are changing the shape of the keys, queries and values
        # to (batchSize, numTokens, numHeads, headDim)
        # so we have to take care that dIn is divisible by numHeads
        assert dIn % self.__numHeads == 0, f"dIn {dIn} is not divisible by numHeads {self.__numHeads}"
        keys = keys.view(batchSize, numTokens, self.__numHeads, headDim)
        queries = queries.view(batchSize, numTokens, self.__numHeads, headDim)
        values = values.view(batchSize, numTokens, self.__numHeads, headDim)

        Logging.GetInstance().Debug(f"keys shape : {keys.shape}")
        # Now we are transposing the keys, queries and values from
        # (batchSize, numTokens, numHeads, headDim) to (batchSize, numHeads, numTokens, headDim)
        # basically we are grouping the keys, queries and values by heads, so that we can run the
        # attention mechanism in parallel for all the heads.
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        Logging.GetInstance().Debug(f"keys shape : {keys.shape}")

        # Attention is calculated using dot product which in a way gives the similarity between
        # querry and keys (which in a way are other elements in the sequence). You may notice that
        # transpose is taken for 2, 3 since we are keeping the batch size and numHeads as is
        # The shape of keys is (batchSize, numHeads, numTokens, headDim)
        # The shape of queries is (batchSize, numHeads, numTokens, headDim)
        # The shape of keysTranspose is (batchSize, numHeads, headDim, numTokens)
        keysTranspose = keys.transpose(2, 3)
        Logging.GetInstance().Debug(f"keyTranspose shape : {keysTranspose.shape}")
        attentionScores = queries @ keysTranspose

        maskBool = self.mask.bool()[:numTokens, :numTokens]
        attentionScores.masked_fill_(maskBool, -torch.inf)

        Logging.GetInstance().Debug(f"attentionScores :\n{attentionScores}")

        attentionWeights = torch.softmax(attentionScores / headDim ** 0.5, dim=-1)
        attentionWeights = self.__dropout(attentionWeights)
        Logging.GetInstance().Debug(f"attentionWeights shape :\n{attentionWeights.shape}")

        # Now we are doing the dot product between attentionWeights and values
        # to get the context vector
        # The shape of attentionWeights is (batchSize, numHeads, numTokens, numToken)
        # The shape of values is (batchSize, numHeads, numTokens, headDim)
        # The shape of contextVector is (batchSize, numHeads, headDim, numTokens)
        Logging.GetInstance().Debug(f"values shape : {values.shape}")

        # Now the grouping again has changed w.r.t number of tokens
        contextVector = (attentionWeights @ values).transpose(1, 2)
        Logging.GetInstance().Debug(f"contextVector : {contextVector}")
        Logging.GetInstance().Debug(f"contextVector contingous : {contextVector.contiguous()}")

        contextVector = contextVector.contiguous().view(batchSize, numTokens, self.__dOut)

        contextVector = self.__wOut(contextVector)

        Logging.GetInstance().Debug(f"contextVector : {contextVector}")
        return contextVector

    def freezeQuery(self):
        for param in self.__wQuery.parameters():
            param.requires_grad = False

    def unfreezeQuery(self):
        for param in self.__wQuery.parameters():
            param.requires_grad = True

    def freezeKey(self):
        for param in self.__wKey.parameters():
            param.requires_grad = False

    def unfreezeKey(self):
        for param in self.__wKey.parameters():
            param.requires_grad = True

    def freezeValue(self):
        for param in self.__wValue.parameters():
            param.requires_grad = False

    def unfreezeValue(self):
        for param in self.__wValue.parameters():
            param.requires_grad = True

class CrossMultiHeadAttention(nn.Module):
    __dIn : int
    __dOut : int
    __contextLength : int
    __numHeads : int
    __wQuery : nn.Linear
    __wKey : nn.Linear
    __wValue : nn.Linear
    __wOut : nn.Linear
    __attentionWeights : torch.Tensor = None

    def __init__(self,
                 dIn : int,
                 dOut : int,
                 contextLength : int,
                 numHeads : int,
                 qkvBias : bool = False,
                 dropout : float = 0.0,
                 device: str = "cpu"):
        super().__init__()

        self.__dIn = dIn
        self.__dOut = dOut
        self.__contextLength = contextLength
        self.__numHeads = numHeads
        self.__wQuery = nn.Linear(dIn, dOut, bias=qkvBias)
        self.__wKey = nn.Linear(dIn, dOut, bias=qkvBias)
        self.__wValue = nn.Linear(dIn, dOut, bias=qkvBias)
        self.__wOut = nn.Linear(dOut, dOut, bias=qkvBias)

        Logging.GetInstance().Debug(f"wQuery : {self.__wQuery}")
        Logging.GetInstance().Debug(f"wKey : {self.__wKey}")
        Logging.GetInstance().Debug(f"wValue : {self.__wValue}")

        self.__dropout = nn.Dropout(dropout)

        # In cross attention we do not use a causal mask, so we do not need to register a mask buffer
        #self.register_buffer("mask", torch.triu(torch.ones(self.__contextLength, self.__contextLength), diagonal=1))

        Logging.GetInstance().Debug(f"SelfAttention initialized with dIn: {dIn}, dOut: {dOut}")

    def getAttentionWeights(self) -> torch.Tensor:
        # This method is used to get the attention weights after the forward pass
        # It is useful for debugging and visualization purposes
        if self.__attentionWeights is None:
            raise ValueError("Attention weights are not available. Please call forward() first.")
        return self.__attentionWeights

    def forward(self, inputs : torch.Tensor, encoderOutputs: torch.Tensor):
        # Inputs received here is in the shape of (batchSize, numTokens, dIn)
        batchSize, numTokens, dIn = inputs.shape
        headDim = self.__dOut // self.__numHeads
        keys = self.__wKey(encoderOutputs)
        queries : torch.tensor = self.__wQuery(inputs)
        values : torch.tensor = self.__wValue(encoderOutputs)

        # In below code we are changing the shape of the keys, queries and values
        # to (batchSize, numTokens, numHeads, headDim)
        # so we have to take care that dIn is divisible by numHeads
        assert dIn % self.__numHeads == 0, f"dIn {dIn} is not divisible by numHeads {self.__numHeads}"
        encLen = encoderOutputs.shape[1]  # encoder sequence length

        keys = keys.view(batchSize, encLen, self.__numHeads, headDim)
        queries = queries.view(batchSize, numTokens, self.__numHeads, headDim)
        values = values.view(batchSize, encLen, self.__numHeads, headDim)

        Logging.GetInstance().Debug(f"keys shape : {keys.shape}")
        # Now we are transposing the keys, queries and values from
        # (batchSize, numTokens, numHeads, headDim) to (batchSize, numHeads, numTokens, headDim)
        # basically we are grouping the keys, queries and values by heads, so that we can run the
        # attention mechanism in parallel for all the heads.
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        Logging.GetInstance().Debug(f"keys shape : {keys.shape}")

        # Attention is calculated using dot product which in a way gives the similarity between
        # querry and keys (which in a way are other elements in the sequence). You may notice that
        # transpose is taken for 2, 3 since we are keeping the batch size and numHeads as is
        # The shape of keys is (batchSize, numHeads, numTokens, headDim)
        # The shape of queries is (batchSize, numHeads, numTokens, headDim)
        # The shape of keysTranspose is (batchSize, numHeads, headDim, numTokens)
        keysTranspose = keys.transpose(2, 3)
        Logging.GetInstance().Debug(f"keyTranspose shape : {keysTranspose.shape}")
        attentionScores = queries @ keysTranspose

        Logging.GetInstance().Debug(f"attentionScores :\n{attentionScores}")

        attentionWeights = torch.softmax(attentionScores / headDim ** 0.5, dim=-1)
        attentionWeights = self.__dropout(attentionWeights)
        Logging.GetInstance().Debug(f"attentionWeights shape :\n{attentionWeights.shape}")
        self.__attentionWeights = attentionWeights

        # Now we are doing the dot product between attentionWeights and values
        # to get the context vector
        # The shape of attentionWeights is (batchSize, numHeads, numTokens, numToken)
        # The shape of values is (batchSize, numHeads, numTokens, headDim)
        # The shape of contextVector is (batchSize, numHeads, headDim, numTokens)
        Logging.GetInstance().Debug(f"values shape : {values.shape}")

        # Now the grouping again has changed w.r.t number of tokens
        contextVector = (attentionWeights @ values).transpose(1, 2)
        Logging.GetInstance().Debug(f"contextVector : {contextVector}")
        Logging.GetInstance().Debug(f"contextVector contingous : {contextVector.contiguous()}")

        contextVector = contextVector.contiguous().view(batchSize, numTokens, self.__dOut)

        contextVector = self.__wOut(contextVector)

        Logging.GetInstance().Debug(f"contextVector : {contextVector}")
        return contextVector

    def freezeQuery(self):
        for param in self.__wQuery.parameters():
            param.requires_grad = False

    def unfreezeQuery(self):
        for param in self.__wQuery.parameters():
            param.requires_grad = True

    def freezeKey(self):
        for param in self.__wKey.parameters():
            param.requires_grad = False

    def unfreezeKey(self):
        for param in self.__wKey.parameters():
            param.requires_grad = True

    def freezeValue(self):
        for param in self.__wValue.parameters():
            param.requires_grad = False

    def unfreezeValue(self):
        for param in self.__wValue.parameters():
            param.requires_grad = True
