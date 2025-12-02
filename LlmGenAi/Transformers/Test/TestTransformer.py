import unittest
import sys

sys.path.append("../../../Common")
sys.path.append("../Src")
sys.path.append("../../Config/Src")

from Logging import Logging
from Logging import LogLevel

import numpy.testing as npt
import torch
import json

from TransformerDecoder import TransformerDecoder
from Config import Config

class TestTransformer(unittest.TestCase):
    def testTransformer(self):
        Logging.GetInstance().Debug("Starting testTransformer\n")

        config = Config("../../Config/Test/TestConfig.json")
        config.loadConfig()


        torch.manual_seed(123)
        batchExample = torch.randn(2, 4, config.getEmbeddingDimension())

        layer = TransformerDecoder(config)
        output: torch.tensor = layer(batchExample)
        Logging.GetInstance().Debug(f"Output: \n{output}")

def main():
    Logging.GetInstance().SetLogLevel(LogLevel.DEBUG)
    unittest.main()

if __name__ == "__main__":
    main()
