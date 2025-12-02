import unittest
import sys

sys.path.append("../../../Common")
sys.path.append("../Src")
sys.path.append("../../Config/Src")
sys.path.append("../../DataPreperation")

from Logging import Logging
from Logging import LogLevel

from Config import Config
from GptImplement import GptImplement

import numpy.testing as npt
import torch
import json
import tiktoken

from TokenEmbedding import GptDataSet
from TokenEmbedding import GptDataLoader

import torch
from transformers import GPT2LMHeadModel

def load_gpt2_weights_into_custom_model(hf_dict, custom_model):
    custom_dict = custom_model.state_dict()

    def log_info(name, hf_shape, custom_shape):
        Logging.GetInstance().Info(f"{name} | HF: {hf_shape} | Custom: {custom_shape}")

    for name, hf_param in hf_dict.items():
        if "attn.c_attn.weight" in name or "attn.c_attn.bias" in name:
            # Get layer number
            layer_idx = int(name.split(".")[2])
            prefix = f"_GptImplement__transformerBlocks.{layer_idx}._Transformer__multiHeadAttention"
            if "weight" in name:
                q, k, v = hf_param.chunk(3, dim=0)
                log_info(f"{name}[Q]", q.shape, custom_dict[f"{prefix}._MultiHeadAttention__wQuery.weight"].shape)
                log_info(f"{name}[K]", k.shape, custom_dict[f"{prefix}._MultiHeadAttention__wKey.weight"].shape)
                log_info(f"{name}[V]", v.shape, custom_dict[f"{prefix}._MultiHeadAttention__wValue.weight"].shape)
                custom_dict[f"{prefix}._MultiHeadAttention__wQuery.weight"].copy_(q)
                custom_dict[f"{prefix}._MultiHeadAttention__wKey.weight"].copy_(k)
                custom_dict[f"{prefix}._MultiHeadAttention__wValue.weight"].copy_(v)
            else:
                q, k, v = hf_param.chunk(3, dim=0)
                log_info(f"{name}[Q]", q.shape, custom_dict[f"{prefix}._MultiHeadAttention__wQuery.bias"].shape)
                log_info(f"{name}[K]", k.shape, custom_dict[f"{prefix}._MultiHeadAttention__wKey.bias"].shape)
                log_info(f"{name}[V]", v.shape, custom_dict[f"{prefix}._MultiHeadAttention__wValue.bias"].shape)
                custom_dict[f"{prefix}._MultiHeadAttention__wQuery.bias"].copy_(q)
                custom_dict[f"{prefix}._MultiHeadAttention__wKey.bias"].copy_(k)
                custom_dict[f"{prefix}._MultiHeadAttention__wValue.bias"].copy_(v)
        elif name.endswith(".weight") or name.endswith(".bias"):
            # Try to match and load
            for key in custom_dict:
                if key.endswith(name.replace("transformer.", "").replace(".", "__")):
                    log_info(name, hf_param.shape, custom_dict[key].shape)
                    custom_dict[key].copy_(hf_param)
                    break


class TestGptModel(unittest.TestCase):

    def TestDummyGptModel(self):

        maxNewTokens = 6

        tokenizer = tiktoken.get_encoding("gpt2")
        #inputText = input("Ask Question : ")
        inputText = "Every effort moves you"

        batch = []
        batch.append(torch.tensor(tokenizer.encode(inputText)))

        batch = torch.stack(batch, dim=0)

        #torch.manual_seed(123)
        configFileName = "../../Config/Test/TestConfig.json"

        model = GptImplement(configFileName)
        tokenBatches = model.generateText(batch, maxNewTokens)

        for tokenBatch in tokenBatches:
            outputText = tokenizer.decode(tokenBatch.tolist())
            Logging.GetInstance().Info(f"Output Text: {outputText}")

    def TestGptLoss(self):
        import tiktoken

        maxNewTokens = 6

        tokenizer = tiktoken.get_encoding("gpt2")
        inputText = "Every effort moves"
        targetText = " effort moves you"

        batch = []
        batch.append(torch.tensor(tokenizer.encode(inputText)))

        batch = torch.stack(batch, dim=0)

        targetTokens = []
        targetTokens.append(torch.tensor(tokenizer.encode(targetText)))
        targetTokens = torch.stack(targetTokens, dim=0)

        torch.manual_seed(123)
        configFileName = "../../Config/Test/TestConfig.json"

        model = GptImplement(configFileName)
        #tokenBatches = model.train(batch, targetTokens)
        logits = model(batch)
        loss = model.getLoss(logits, targetTokens)
        Logging.GetInstance().Info(f"Loss: {loss.item()}")
        self.assertAlmostEqual(loss.item(), 10.52, places=2)
        perplexity = model.getPerplexity(loss)
        Logging.GetInstance().Info(f"Perplexity: {perplexity}")
        self.assertAlmostEqual(perplexity.item(), 37081.48, places=2)

    def TestGptLossWithRealData(self):
        fileName = "../../../Learning/LearningData/the-verdict.txt"

        device = torch.device("cpu" if torch.mps.is_available() else "cpu")
        Logging.GetInstance().Info(f"Device: {device}")

        with open(fileName, "r") as file:
            inputText = file.read()
        gptDataSet = GptDataSet()
        gptDataSet.setRawText(inputText)
        gptDataSet.tokenize()
        gptDataSet.buildInputTargetPair()
        gptDataLoader = GptDataLoader(gptDataSet, batchSize=2, shuffle=True)
        dataLoader = gptDataLoader.getDataLoader()
        torch.manual_seed(123)
        configFileName = "../../Config/Test/TestConfig.json"
        model = GptImplement(configFileName)
        model.to(device)
        totalLoss = 0.0
        totalPerplexity = 0.0
        totalCount = 0
        for inputIds, targetIds in dataLoader:
            inputIds = inputIds.to(device)
            targetIds = targetIds.to(device)
            Logging.GetInstance().Debug(f"Input IDs shape: {inputIds.shape}, Target IDs shape: {targetIds.shape}")
            logits = model(inputIds)
            loss = model.getLoss(logits, targetIds)
            totalLoss += loss.item()
            perplexity = model.getPerplexity(loss)
            totalPerplexity += perplexity.item()
            totalCount += 1
            self.assertGreater(perplexity.item(), 1.0)
            Logging.GetInstance().Info(f"Count: {totalCount}")

        averageLoss = totalLoss / totalCount
        averagePerplexity = totalPerplexity / totalCount
        Logging.GetInstance().Info(f"Average Loss: {averageLoss}")
        Logging.GetInstance().Info(f"Average Perplexity: {averagePerplexity}")

    def testGptTraining(self):
        import os
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        fileName = "../../../Learning/LearningData/dialogues_text_cleaned.txt"

        device = torch.device("cpu" if torch.mps.is_available() else "cpu")
        Logging.GetInstance().Info(f"Device: {device}")

        with open(fileName, "r") as file:
            inputText = file.read()

        trainRatio = 1
        splitIdx = int(trainRatio * len(inputText))
        trainData = inputText[:splitIdx]

        gptDataSet = GptDataSet()
        gptDataSet.setRawText(trainData)
        gptDataSet.tokenize()
        gptDataSet.buildInputTargetPair()
        gptDataLoader = GptDataLoader(gptDataSet, batchSize=2, shuffle=True, numWorkers=10)
        dataLoader = gptDataLoader.getDataLoader()
        #torch.manual_seed(123)
        configFileName = "../../Config/Test/TestConfig.json"
        model = GptImplement(configFileName, numEpochs=3)

        tokenizer = tiktoken.get_encoding("gpt2")
        inputText = input("Ask Question : ")
        #inputText = "Every effort moves you"

        batch = []
        batch.append(torch.tensor(tokenizer.encode(inputText)).to(torch.device("cpu")))

        batch = torch.stack(batch, dim=0)

        maxNewTokens = 6

        tokenBatches = model.generateText(batch, maxNewTokens)
        for tokenBatch in tokenBatches:
            outputText = tokenizer.decode(tokenBatch.tolist())
            Logging.GetInstance().Info(f"Output Text: {outputText}")

        Logging.GetInstance().Info("I think I spoke some gibberish, let me train myself")

        model.setDataLoader(gptDataLoader)
        model.train(saveFileName="../Parameters/LocalGpt2_DailyDialogue.pth", loadFileName="../Parameters/LocalGpt2_DailyDialogue.pth", isCheckpoint=False)

        tokenizer = tiktoken.get_encoding("gpt2")
        inputText = input("Ask Question : ")
        #inputText = "Every effort moves you"

        batch = []
        batch.append(torch.tensor(tokenizer.encode(inputText)).to(torch.device("cpu")))

        batch = torch.stack(batch, dim=0)

        tokenBatches = model.generateText(batch, maxNewTokens)
        for tokenBatch in tokenBatches:
            outputText = tokenizer.decode(tokenBatch.tolist())
            Logging.GetInstance().Info(f"Output Text: {outputText}")

    def testGptWithLocalPretrainedWeights(self):

        device = torch.device("cpu" if torch.mps.is_available() else "cpu")
        Logging.GetInstance().Info(f"Device: {device}")
        #torch.set_num_threads(12)
        #torch.set_num_interop_threads(12)

        configFileName = "../../Config/Test/TestConfig.json"
        model = GptImplement(configFileName, numEpochs=3)
        model.loadParameters("../Parameters/LocalGpt2_DailyDialogue.pth")


        tokenizer = tiktoken.get_encoding("gpt2")
        inputText = input("Ask Question : ")
        #inputText = "Every effort moves you"

        batch = []
        batch.append(torch.tensor(tokenizer.encode(inputText)).to(torch.device("cpu")))

        batch = torch.stack(batch, dim=0)

        maxNewTokens = 50

        tokenBatches = model.generateText(batch, maxNewTokens)
        for tokenBatch in tokenBatches:
            outputText = tokenizer.decode(tokenBatch.tolist())
            Logging.GetInstance().Info(f"Output Text: {outputText}")

    def TestGptWithPretrainedWeights(self):

        device = torch.device("cpu" if torch.mps.is_available() else "cpu")
        Logging.GetInstance().Info(f"Device: {device}")
        #torch.set_num_threads(12)
        #torch.set_num_interop_threads(12)

        configFileName = "../../Config/Test/TestConfig.json"
        model = GptImplement(configFileName, numEpochs=3)
        #model.loadParameters("../Parameters/gpt2_1558M.pth")

        # Load Hugging Face model
        hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
        hf_state = hf_model.state_dict()

        # Then call this helper with your model instance:
        load_gpt2_weights_into_custom_model(hf_state, model)


        tokenizer = tiktoken.get_encoding("gpt2")
        inputText = input("Ask Question : ")
        #inputText = "Every effort moves you"

        batch = []
        batch.append(torch.tensor(tokenizer.encode(inputText)).to(torch.device("cpu")))

        batch = torch.stack(batch, dim=0)

        maxNewTokens = 6

        tokenBatches = model.generateText(batch, maxNewTokens)
        for tokenBatch in tokenBatches:
            outputText = tokenizer.decode(tokenBatch.tolist())
            Logging.GetInstance().Info(f"Output Text: {outputText}")

def main():
    Logging.GetInstance().SetLogLevel(LogLevel.INFO)
    torch.set_num_interop_threads(12)
    torch.set_num_threads(12)
    unittest.main()

if __name__ == "__main__":
    main()