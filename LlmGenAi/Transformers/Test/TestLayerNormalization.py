#######################################################################
# File - TestLayerNormalization.py
# Author - Mayank Arora
# Description - This is file tests the implementation of layer
#               normalization.
#######################################################################
import unittest
import sys

sys.path.append("../../../Common")
sys.path.append("../Src")
sys.path.append("../../Config/Src")

from Logging import Logging
from Logging import LogLevel

from LayerNormalization import LayerNormalization
from Config import Config

import numpy.testing as npt
import torch
import json

class TestLayerNormalization(unittest.TestCase):
    def testDummyLayerNormalization(self):
        Logging.GetInstance().Debug("Starting testDummyLayerNormalization\n")

        torch.manual_seed(123)
        batchExample = torch.randn(2, 5)
        layer = torch.nn.Sequential(torch.nn.Linear(5, 6), torch.nn.ReLU())
        output : torch.tensor = layer(batchExample)

        Logging.GetInstance().Debug(f"Output: {output}")

        mean = output.mean(dim=-1, keepdim=True)
        var = output.var(dim=-1, unbiased=False, keepdim=True)

        Logging.GetInstance().Debug(f"\nMean: {mean},\nVariance: {var}")

        outputNormalized = (output - mean) / torch.sqrt(var)

    def testLayerNormalization(self):
        Logging.GetInstance().Debug("Starting testLayerNormalization\n")

        torch.manual_seed(123)
        batchExample = torch.randn(2, 5)

        with open("../../Config/Test/TestConfig.json", "r") as file:
            testConfigJson = json.load(file)

            gptConfigType = "layerNormConfig"

            testGptConfig = testConfigJson.get(gptConfigType)

            config = Config("../../Config/Test/TestConfig.json")
            config.loadConfig(gptConfigType)

            layer = LayerNormalization(config)
            output: torch.tensor = layer(batchExample)
            Logging.GetInstance().Debug(f"Output: \n{output}")


def main():
    Logging.GetInstance().SetLogLevel(LogLevel.DEBUG)
    unittest.main()
if __name__ == "__main__":
    main()