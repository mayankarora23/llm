#######################################################################
# File - TestConfig.py
# Author - Mayank Arora
# Description - This test file to test the implementation of extracting
#               and holding the configuration for the transformers.
#######################################################################
import unittest
import sys

sys.path.append("../../../Common")
sys.path.append("../Src")

from Logging import Logging
from Logging import LogLevel
import numpy.testing as npt

from Config import Config

import json

class TestGptConfig(unittest.TestCase):
    def test_ConfigInitialization(self):
        Logging.GetInstance().Debug("Starting test_ConfigInitialization\n")

        config = Config("./TestConfig.json")
        config.loadConfig()

        self.assertEqual(config.getVocabSize(), 50257)
        self.assertEqual(config.getContextLength(), 1024)
        self.assertEqual(config.getEmbeddingDimension(), 768)
        self.assertEqual(config.getAttentionHeads(), 12)
        self.assertEqual(config.getNumLayers(), 12)
        self.assertAlmostEqual(config.getDropoutRate(), 0.1, places=2)
        self.assertFalse(config.useQueryKeyValueBias())

        Logging.GetInstance().Debug("Finished test_ConfigInitialization\n")

    def test_DefaultConfig(self):
        Logging.GetInstance().Debug("Starting test_DefaultConfig\n")

        with open("./TestConfig.json", "r") as file:
            testConfigJson = json.load(file)

            gptConfigType = testConfigJson.get("useGptConfig")

            testGptConfig = testConfigJson.get(gptConfigType)

            config = Config("./TestConfig.json")
            config.loadConfig()

            self.assertEqual(config.getVocabSize(), testGptConfig.get("vocabularySize"))
            self.assertEqual(config.getContextLength(), testGptConfig.get("contextLength"))
            self.assertEqual(config.getEmbeddingDimension(), testGptConfig.get("embeddingDimension"))
            self.assertEqual(config.getAttentionHeads(), testGptConfig.get("attentionHeads"))
            self.assertEqual(config.getNumLayers(), testGptConfig.get("numLayers"))
            self.assertAlmostEqual(config.getDropoutRate(), testGptConfig.get("dropoutRate"), places=2)
            self.assertEqual(config.useQueryKeyValueBias(), testGptConfig.get("useQueryKeyValueBias"))

        Logging.GetInstance().Debug("Finished test_DefaultConfig\n")

    def test_124MConfig(self):
        Logging.GetInstance().Debug("Starting test_124MConfig\n")

        with open("./TestConfig.json", "r") as file:
            testConfigJson = json.load(file)

            gptConfigType = "gptConfig124M"

            testGptConfig = testConfigJson.get(gptConfigType)

            config = Config("./TestConfig.json")
            config.loadConfig(gptConfigType)

            self.assertEqual(config.getVocabSize(), testGptConfig.get("vocabularySize"))
            self.assertEqual(config.getContextLength(), testGptConfig.get("contextLength"))
            self.assertEqual(config.getEmbeddingDimension(), testGptConfig.get("embeddingDimension"))
            self.assertEqual(config.getAttentionHeads(), testGptConfig.get("attentionHeads"))
            self.assertEqual(config.getNumLayers(), testGptConfig.get("numLayers"))
            self.assertAlmostEqual(config.getDropoutRate(), testGptConfig.get("dropoutRate"), places=2)
            self.assertEqual(config.useQueryKeyValueBias(), testGptConfig.get("useQueryKeyValueBias"))

        Logging.GetInstance().Debug("Finished test_124MConfig\n")

    def test_355MConfig(self):
        Logging.GetInstance().Debug("Starting test_355MConfig\n")

        with open("./TestConfig.json", "r") as file:
            testConfigJson = json.load(file)

            gptConfigType = "gptConfig355M"

            testGptConfig = testConfigJson.get(gptConfigType)

            config = Config("./TestConfig.json")
            config.loadConfig(gptConfigType)

            self.assertEqual(config.getVocabSize(), testGptConfig.get("vocabularySize"))
            self.assertEqual(config.getContextLength(), testGptConfig.get("contextLength"))
            self.assertEqual(config.getEmbeddingDimension(), testGptConfig.get("embeddingDimension"))
            self.assertEqual(config.getAttentionHeads(), testGptConfig.get("attentionHeads"))
            self.assertEqual(config.getNumLayers(), testGptConfig.get("numLayers"))
            self.assertAlmostEqual(config.getDropoutRate(), testGptConfig.get("dropoutRate"), places=2)
            self.assertEqual(config.useQueryKeyValueBias(), testGptConfig.get("useQueryKeyValueBias"))

        Logging.GetInstance().Debug("Finished test_355MConfig\n")

    def test_774Config(self):
        Logging.GetInstance().Debug("Starting test_774Config\n")

        with open("./TestConfig.json", "r") as file:
            testConfigJson = json.load(file)

            gptConfigType = "gptConfig774M"

            testGptConfig = testConfigJson.get(gptConfigType)

            config = Config("./TestConfig.json")
            config.loadConfig(gptConfigType)

            self.assertEqual(config.getVocabSize(), testGptConfig.get("vocabularySize"))
            self.assertEqual(config.getContextLength(), testGptConfig.get("contextLength"))
            self.assertEqual(config.getEmbeddingDimension(), testGptConfig.get("embeddingDimension"))
            self.assertEqual(config.getAttentionHeads(), testGptConfig.get("attentionHeads"))
            self.assertEqual(config.getNumLayers(), testGptConfig.get("numLayers"))
            self.assertAlmostEqual(config.getDropoutRate(), testGptConfig.get("dropoutRate"), places=2)
            self.assertEqual(config.useQueryKeyValueBias(), testGptConfig.get("useQueryKeyValueBias"))

        Logging.GetInstance().Debug("Finished test_774Config\n")

    def test_1558Config(self):
        Logging.GetInstance().Debug("Starting test_1558Config\n")

        with open("./TestConfig.json", "r") as file:
            testConfigJson = json.load(file)

            gptConfigType = "gptConfig1558M"

            testGptConfig = testConfigJson.get(gptConfigType)

            config = Config("./TestConfig.json")
            config.loadConfig(gptConfigType)

            self.assertEqual(config.getVocabSize(), testGptConfig.get("vocabularySize"))
            self.assertEqual(config.getContextLength(), testGptConfig.get("contextLength"))
            self.assertEqual(config.getEmbeddingDimension(), testGptConfig.get("embeddingDimension"))
            self.assertEqual(config.getAttentionHeads(), testGptConfig.get("attentionHeads"))
            self.assertEqual(config.getNumLayers(), testGptConfig.get("numLayers"))
            self.assertAlmostEqual(config.getDropoutRate(), testGptConfig.get("dropoutRate"), places=2)
            self.assertEqual(config.useQueryKeyValueBias(), testGptConfig.get("useQueryKeyValueBias"))

        Logging.GetInstance().Debug("Finished test_1558Config\n")

def main():
    Logging.GetInstance().SetLogLevel(LogLevel.INFO)
    unittest.main()

if __name__ == "__main__":
    main()