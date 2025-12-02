import unittest
import sys

sys.path.append("../../../Common")
sys.path.append("../Src")

from Logging import Logging
from Logging import LogLevel
import numpy.testing as npt

import torch

import Attention 

def testMultiHeadedAttention():
    torch.manual_seed(123)

#    inputsList = [
#    [0.43, 0.15, 0.89], #your
#    [0.55, 0.87, 0.66], #journey
#    [0.57, 0.85, 0.64], #starts
#    [0.22, 0.58, 0.33], #with
#    [0.77, 0.25, 0.10], #one
#    [0.05, 0.80, 0.55], #step
#]

    inputsList = [
    [0.43, 0.15, 0.89, 0.55, 0.87, 0.66, 0.77, 0.88],  # Row 1
    [0.57, 0.85, 0.64, 0.22, 0.58, 0.33, 0.55, 0.44],  # Row 2
    [0.77, 0.25, 0.10, 0.05, 0.80, 0.55, 0.33, 0.22]]  # Row 3

    inputs = torch.Tensor(inputsList)



    #dIn = inputs.shape[1]
    #dOut = inputs.shape[1]
    #dOut = 2

    batch = torch.stack((inputs, inputs), dim=0)
    batchSize, contextLength, dIn = batch.shape

    dOut = dIn
    Logging.GetInstance().Info(f"dIn = {dIn}, dOut = {dOut}")

    #contextLength = batch.shape[1]
    Logging.GetInstance().Info(f"batch = {batch}")
    Logging.GetInstance().Info(f"batch shape = {batch.shape}")
    Logging.GetInstance().Debug(f"contextLength = {contextLength}")

    #causalAttentionMask = CausalAttentionMask(dIn, dOut, contextLength, 0.0)

    #maskedMultiHeadAttention = MultiHeadAttentionWrapper(dIn, dOut, contextLength, 2, False, 0.0)
    maskedMultiHeadAttention = Attention.MaskedMultiHeadAttention(dIn, dOut, contextLength, 2, False, 0.0)
    maskedMultiHeadAttention(batch)

    #contextVector = maskedMultiHeadAttention(batch)

    #Logging.GetInstance().Info(f"contextVector : {contextVector}")

def main():
    Logging.GetInstance().SetLogLevel(LogLevel.DEBUG)
    Logging.GetInstance().Debug("Starting test - CausalAttentionMask")
    testMultiHeadedAttention()
    Logging.GetInstance().Debug("Ending test - CausalAttentionMask")

if __name__ == "__main__":
    main()
