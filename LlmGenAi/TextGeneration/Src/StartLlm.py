#######################################################################
# File - MyModelImplement.py
# Author - Mayank Arora
# Description - This file contains the implementation of GPT model
#               based on GPT2 architecture
#######################################################################

import sys
from typing import Dict
from typing import Callable
#import tiktoken
from transformers import AutoTokenizer
from pynput import keyboard

import torch

import torch
import torch.nn as nn

sys.path.append("../../../Common")
sys.path.append("../../Config/Src")
sys.path.append("../../Transformers/Src")
sys.path.append("../../DataPreperation")

from Logging import Logging
from Logging import LogLevel

from Config import Config
from MyModelImplement import MyModel as llm
from ModelEncaps import ModelEncaps as llmEncaps
from FineTuningLora import FineTuningLora as lora
from Config import Config

from torch.utils.data import DistributedSampler

from TokenEmbedding import EncoderDecoderDataSet
from TokenEmbedding import EncoderDecoderDataLoader

from abc import abstractmethod

class LlmRunType(enumerate):
    TRAIN = 0
    RUN = 1
    EVALUATE = 2
    FINETUNE = 3
    UNKNOWN = 4

class CmdArgumentParserBase:
    _startLlm : 'StartLlm' = None
    @abstractmethod
    def parseAndInitFromCmdArguments(self, startLlm: 'StartLlm') -> tuple[bool, Callable]:
        raise NotImplementedError("Subclasses should implement this method")

class HelpCmdArgumentParser(CmdArgumentParserBase):
    def handleHelpCmdArgumentParser(self) -> bool:
        Logging.GetInstance().SetDetailedLogNeeded(False)
        terminateExecution = True
        Logging.GetInstance().Info("Usage: python StartLlm.py -options")
        Logging.GetInstance().Info("Options:")
        Logging.GetInstance().Info("--config <config_file> : Path to the configuration file")
        Logging.GetInstance().Info("--device <device> : Device to run the model on (default: cpu)")
        Logging.GetInstance().Info("--numEpochs <num> : Number of epochs to train the model (default: 4)")
        Logging.GetInstance().Info("--train <data set> : Path to the training data set")
        Logging.GetInstance().Info("--runllm : Run the LLM after initialization")
        Logging.GetInstance().Info("--finetune : Fine tune the LLM on the specified data set")
        Logging.GetInstance().Info("--saveparams <parameter_file> : Save the parameters to the specified file")
        Logging.GetInstance().Info("--loadparams <parameter_file> : Load the parameters from the specified file")
        Logging.GetInstance().Info("--help : Show this help message")
        Logging.GetInstance().SetDetailedLogNeeded(True)
        return terminateExecution

    def parseAndInitFromCmdArguments(self, startLlm: 'StartLlm') -> tuple[bool, Callable]:
        self._startLlm = startLlm
        isParameterExpected = False
        return isParameterExpected, self.handleHelpCmdArgumentParser

class ConfigCmdArgumentParser(CmdArgumentParserBase):
    def handleConfigCmdArgumentParser(self, configFileName: str) -> bool:
        self._startLlm.setConfigFileName(configFileName)
        Logging.GetInstance().Info(f"Configuration file set to: {configFileName}")
        return False

    def parseAndInitFromCmdArguments(self, startLlm: 'StartLlm') -> tuple[bool, Callable]:
        self._startLlm = startLlm
        isParameterExpected = True
        return isParameterExpected, self.handleConfigCmdArgumentParser

class DeviceCmdArgumentParser(CmdArgumentParserBase):
    def handleDeviceCmdArgumentParser(self, device: str) -> bool:
        self._startLlm.setDevice(device)
        Logging.GetInstance().Info(f"Device set to: {device}")
        return False

    def parseAndInitFromCmdArguments(self, startLlm: 'StartLlm') -> tuple[bool, Callable]:
        self._startLlm = startLlm
        isParameterExpected = True
        return isParameterExpected, self.handleDeviceCmdArgumentParser

class NumEpochsCmdArgumentParser(CmdArgumentParserBase):
    def handleNumEpochsCmdArgumentParser(self, numEpochs: str) -> bool:
        try:
            numEpochsInt = int(numEpochs)
            self._startLlm.setNumEpochs(numEpochsInt)
            Logging.GetInstance().Info(f"Number of epochs set to: {numEpochsInt}")
        except ValueError:
            Logging.GetInstance().Error("Invalid number of epochs. Please provide an integer value.")
            return True
        return False

    def parseAndInitFromCmdArguments(self, startLlm: 'StartLlm') -> tuple[bool, Callable]:
        self._startLlm = startLlm
        isParameterExpected = True
        return isParameterExpected, self.handleNumEpochsCmdArgumentParser

class RunLlmCmdArgumentParser(CmdArgumentParserBase):
    def handleRunLlmCmdArgumentParser(self) -> bool:
        self._startLlm.setLlmRunType(LlmRunType.RUN)
        Logging.GetInstance().Info("Running LLM after initialization.")
        return False

    def parseAndInitFromCmdArguments(self, startLlm: 'StartLlm') -> tuple[bool, Callable]:
        self._startLlm = startLlm
        isParameterExpected = False
        return isParameterExpected, self.handleRunLlmCmdArgumentParser

class FineTuneLlmCmdArgumentParser(CmdArgumentParserBase):
    def handleFineTuneLlmCmdArgumentParser(self, dataSet: str) -> bool:
        self._startLlm.setLlmRunType(LlmRunType.FINETUNE)
        self._startLlm.setTrainingDataSet(dataSet)
        Logging.GetInstance().Info(f"Fine-tuning LLM with data set: {dataSet}")
        return False

    def parseAndInitFromCmdArguments(self, startLlm: 'StartLlm') -> tuple[bool, Callable]:
        self._startLlm = startLlm
        isParameterExpected = True
        return isParameterExpected, self.handleFineTuneLlmCmdArgumentParser

class TrainLlmCmdArgumentParser(CmdArgumentParserBase):
    def handleTrainLlmCmdArgumentParser(self, dataSet: str) -> bool:
        self._startLlm.setLlmRunType(LlmRunType.TRAIN)
        self._startLlm.setTrainingDataSet(dataSet)
        Logging.GetInstance().Info(f"Training LLM with data set: {dataSet}")
        return False

    def parseAndInitFromCmdArguments(self, startLlm: 'StartLlm') -> tuple[bool, Callable]:
        self._startLlm = startLlm
        isParameterExpected = True
        return isParameterExpected, self.handleTrainLlmCmdArgumentParser

class LoadParamsCmdArgumentParser(CmdArgumentParserBase):
    def handleLoadParamsCmdArgumentParser(self, paramsFileName: str) -> bool:
        if not paramsFileName:
            Logging.GetInstance().Error("Parameters file name is missing. Please provide a valid file name.")
            return True
        Logging.GetInstance().Info(f"Parameters file set to: {paramsFileName}")
        self._startLlm.setLoadParamsFileName(paramsFileName)
        # Logic to load parameters from the file can be added here
        return False

    def parseAndInitFromCmdArguments(self, startLlm: 'StartLlm') -> tuple[bool, Callable]:
        self._startLlm = startLlm
        isParameterExpected = True
        return isParameterExpected, self.handleLoadParamsCmdArgumentParser

class SaveParamsCmdArgumentParser(CmdArgumentParserBase):
    def handleSaveParamsCmdArgumentParser(self, paramsFileName: str) -> bool:
        if not paramsFileName:
            Logging.GetInstance().Error("Parameters file name is missing. Please provide a valid file name.")
            return True
        Logging.GetInstance().Info(f"Parameters will be saved to: {paramsFileName}")
        self._startLlm.setSaveParamsFileName(paramsFileName)
        # Logic to save parameters to the file can be added here
        return False

    def parseAndInitFromCmdArguments(self, startLlm: 'StartLlm') -> tuple[bool, Callable]:
        self._startLlm = startLlm
        isParameterExpected = True
        return isParameterExpected, self.handleSaveParamsCmdArgumentParser

class StartLlm:
    __model: llmEncaps
    __device: str
    __configFileName: str = ""
    __numEpochs: int
    __cmdArgumentParser: Dict[str, CmdArgumentParserBase] = {}
    __llmRunType: LlmRunType = LlmRunType.UNKNOWN
    __trainingDataSet: str = ""
    __loadParamsFileName: str = ""
    __saveParamsFileName: str = ""
    __keysPressed = set()
    __config: Config = None
    __terminateExecution: bool = False

    def __init__(self, configFileName, device: str = "cpu"):

        self.__device = device
        self.__numEpochs = 4
        self.__configFileName = configFileName
        self.__llmRunType = LlmRunType.UNKNOWN

        self.__cmdArgumentParser["--help"] = HelpCmdArgumentParser()
        self.__cmdArgumentParser["--config"] = ConfigCmdArgumentParser()
        self.__cmdArgumentParser["--device"] = DeviceCmdArgumentParser()
        self.__cmdArgumentParser["--numEpochs"] = NumEpochsCmdArgumentParser()
        self.__cmdArgumentParser["--runllm"] = RunLlmCmdArgumentParser()
        self.__cmdArgumentParser["--train"] = TrainLlmCmdArgumentParser()
        self.__cmdArgumentParser["--finetune"] = FineTuneLlmCmdArgumentParser()
        self.__cmdArgumentParser["--loadparams"] = LoadParamsCmdArgumentParser()
        self.__cmdArgumentParser["--saveparams"] = SaveParamsCmdArgumentParser()

        if len(sys.argv) == 1:
            Logging.GetInstance().Info(f"No command line arguments provided. "
                                       f"Using configuration from config file: {configFileName}")
            self.__initFromConfig()
            terminateExecution = False
        else:
            terminateExecution = self.__parseAndInitFromCmdArguments()

        if terminateExecution:
            self.__terminateExecution = True
            return

        #model = llm(configFileName=self.__configFileName,
        #            device=self.__device)
        model = llmEncaps(config=self.__config, device=self.__device)
        model.initDevices()
        #if self.__device == "cuda":
        #    model = torch.nn.DataParallel(model, device_ids=[0, 1])
        #    model = model.cuda()
        self.__model = model
        #self.__config = self.getModel().getConfig()

        Logging.GetInstance().Info(f"Model initialized with config: {configFileName} on device: {self.__device}")

    def __initAndGetDistributedDataSampler(self, dataSet: EncoderDecoderDataSet):
        import os
        sampler = None
        if self.__config.getLlmStartConfig().useDdp():
            rank = int(os.environ["RANK"])
            worldSize = int(os.environ["WORLD_SIZE"])
            sampler = DistributedSampler(dataSet, num_replicas=worldSize, rank=rank, shuffle=True)
        return sampler

    def getTerminateExecution(self) -> bool:
        return self.__terminateExecution

    def ctrlQPressed(self):
        Logging.GetInstance().Info("Ctrl+Q pressed. Exiting...")
        sys._exit(0)

    def getModel(self) -> llm:
        if self.__device == "cuda":
            return self.__model.getModel().module
        else:
            return self.__model.getModel()
        #return self.__model.module if isinstance(self.__model, torch.nn.DataParallel) else self.__model

    def getDevice(self) -> str:
        return self.__device

    def setDevice(self, device: str):
        self.__device = device

    def getNumEpochs(self) -> int:
        return self.__numEpochs

    def setNumEpochs(self, numEpochs: int):
        self.__numEpochs = numEpochs

    def getConfigFileName(self) -> str:
        return self.__configFileName

    def setConfigFileName(self, configFileName: str):
        self.__configFileName = configFileName

    def getLlmRunType(self) -> LlmRunType:
        return self.__llmRunType

    def setLlmRunType(self, llmRunType: LlmRunType):
        self.__llmRunType = llmRunType

    def getTrainingDataSet(self) -> str:
        return self.__trainingDataSet

    def setTrainingDataSet(self, trainingDataSet: str):
        self.__trainingDataSet = trainingDataSet

    def getLoadParamsFileName(self) -> str:
        return self.__loadParamsFileName

    def setLoadParamsFileName(self, paramsFileName: str):
        self.__loadParamsFileName = paramsFileName

    def getSaveParamsFileName(self) -> str:
        return self.__saveParamsFileName

    def setSaveParamsFileName(self, paramsFileName: str):
        self.__saveParamsFileName = paramsFileName

    def __initFromConfig(self):
        self.__config = Config(configFileName)
        self.__config.loadConfig()

        # Initializing configurations from config file
        self.__device = self.__config.getLlmStartConfig().getDevice()
        self.__numEpochs = self.__config.getLlmStartConfig().getNumEpochs()

        if self.__config.getLlmStartConfig().isTraining() == True:
            self.__llmRunType = LlmRunType.TRAIN
        elif self.__config.getLlmStartConfig().isFineTuning() == True:
            self.__llmRunType = LlmRunType.FINETUNE
        elif self.__config.getLlmStartConfig().runLlm() == True:
            self.__llmRunType = LlmRunType.RUN
        else:
            self.__llmRunType = LlmRunType.UNKNOWN

        self.__trainingDataSet = self.__config.getLlmStartConfig().getDataSetPath()
        if self.__config.getLlmStartConfig().useLoadParams() == True:
            self.__loadParamsFileName = self.__config.getLlmStartConfig().getLoadParamsFileName()
        else:
            self.__loadParamsFileName = ""
        self.__saveParamsFileName = self.__config.getLlmStartConfig().getSaveParamsFileName()

    def __parseAndInitFromCmdArguments(self) -> bool:
        terminateExectution = False
        isParameterExpected = False
        functionRef: Callable = None

        if len(sys.argv) < 2:
            raise ValueError(f"run parameters are missing. For more informaion run -\npython {sys.argv[0]} --help")

        for arg in sys.argv[1:]:
            Logging.GetInstance().Debug(f"argument: {arg}, isParameterExpected: {isParameterExpected}, functionRef: {functionRef}")
            if not isParameterExpected:
                parser = self.__cmdArgumentParser[arg]
            else:
                isParameterExpected = False
                if functionRef is not None:
                    terminateExectution = functionRef(arg)
                    functionRef = None
                    if terminateExectution:
                        return True
            if parser:
                isParameterExpected, functionRef  = parser.parseAndInitFromCmdArguments(self)
                parser = None
                if not isParameterExpected:
                    terminateExectution = functionRef()
                    functionRef = None
                    if terminateExectution:
                        return True
        return terminateExectution

    def __onKeyPress(self, key):
        try:
            # Add key to the set of pressed keys
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.__keysPressed.add('ctrl')
            elif hasattr(key, 'char') and key.char.lower() == 'q':
                if 'ctrl' in self.__keysPressed:
                    self.ctrlQPressed()
        except AttributeError:
            pass  # Ignore special keys


    def __filterPostProcessedText(self, inputText: str, outputText: str) -> str:
        import re
        # Remove the input text from the output text
        if inputText in outputText:
            outputText = outputText.replace(inputText, "")

        # Remove any leading or trailing whitespace
        outputText = outputText.strip()
        outputText = outputText.lstrip("?!. ")

        # Replace multiple continuous spaces with single space
        outputText = re.sub(r'\s+', ' ', outputText)

        return outputText

    def __runLlm(self):
        #if self.__loadParamsFileName == "":
            #Logging.GetInstance().Error("Parameters file name is missing. Please provide a valid file name using --params <file_name>.")
            #return False
        #try:
            listener = keyboard.Listener(on_press=self.__onKeyPress)
            listener.start()
            self.getModel().loadCheckpoint(self.__loadParamsFileName, 0)
            #tokenizer = tiktoken.get_encoding("gpt2")
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.getModel().setTokenizer(tokenizer)
            while True:
                inputText = input("\n\nUser > ").strip()

                if not inputText:
                    print("⚠️  Empty input, please enter a prompt.")
                    continue

                tokenIds = tokenizer.encode(inputText)
                if len(tokenIds) == 0:
                    print("⚠️  Tokenizer produced empty output. Try different input.")
                    continue

                batch = []
                batch.append(torch.tensor(tokenizer.encode(inputText)))

                batch = torch.stack(batch, dim=0)

                maxNewTokens = 60

                self.getModel().eval()
                tokenBatches = self.getModel().generateText(batch, maxNewTokens)

                outputText = ""

                for tokenBatch in tokenBatches:
                    #if 50256 in tokenBatch:
                    #    tokenBatch = tokenBatch[:tokenBatch.tolist().index(50256)]
                    #    outputText = tokenizer.decode(tokenBatch.tolist())
                        #outputText = self.__filterPostProcessedText(inputText, outputText)
                    #    break
                    #if 50257 in tokenBatch or 50258 in tokenBatch:
                    #    continue
                    #else:
                    #    outputText = tokenizer.decode(tokenBatch.tolist())
                        #outputText = self.__filterPostProcessedText(inputText, outputText)
                    vocabSize = 32128
                    outputText = tokenizer.decode([
                        tok for tok in tokenBatch.tolist() if tok not in (vocabSize + 1, 0, vocabSize + 2, vocabSize + 3, vocabSize + 4)], skip_special_tokens=True)
                    outputText = self.__filterPostProcessedText(inputText, outputText)

                Logging.GetInstance().SetDetailedLogNeeded(False)
                Logging.GetInstance().Info(f"\nBot > {outputText}")
                Logging.GetInstance().SetDetailedLogNeeded(True)

        #except Exception as e:
            #Logging.GetInstance().Error(f"Error running LLM: {str(e)}")

    def __initializeParameters(self, module):
        import math
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Normal(0, 0.02)
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def __trainLlm(self):
        if self.__trainingDataSet == "":
            Logging.GetInstance().Error("Training data set is missing. Please provide a valid data set using --train <data set>.")
            return False

        if self.__saveParamsFileName == "":
            Logging.GetInstance().Error("Parameters file name is missing. Please provide a valid file name using --saveparams <file_name>.")
            return False

        #try:
        with open(self.__trainingDataSet, "r") as file:
            inputText = file.read()

            trainRatio = 1
            batchSize = self.__config.getLearningConfig().getBatchSize()
            splitIdx = int(trainRatio * len(inputText))
            trainData = inputText[:splitIdx]

            dataSet = EncoderDecoderDataSet(config=self.__config, device=self.__device)
            dataSet.setRawText(trainData)
            #dataSet.tokenize()
            dataSet.buildInputTargetPair(maxLength = self.__config.getContextLength())

            dataSampler = self.__initAndGetDistributedDataSampler(dataSet)
            gptDataLoader = EncoderDecoderDataLoader(dataSet,
                                                     batchSize=batchSize,
                                                     shuffle=True,
                                                     numWorkers=10,
                                                     sampler=dataSampler)
            dataLoader = gptDataLoader.getDataLoader()

            totalParameters = sum(p.numel() for p in self.getModel().parameters() if p.requires_grad)
            Logging.GetInstance().Info(f"Total trainable parameters: {totalParameters}, batch size: {batchSize}, number of batches per epoch: {len(dataLoader)}")

            self.__model.setTokenizer(dataSet.getTokenizer())
            self.__model.setDataLoader(gptDataLoader)
            self.__model.setNumEpochs(self.__numEpochs)

            self.getModel().apply(lambda m: self.__initializeParameters(m))

            self.__model.runTraining(saveFileName=self.__saveParamsFileName, loadFileName=self.__loadParamsFileName, isCheckpoint=True)

        #except Exception as e:
        #    Logging.GetInstance().Error(f"Error training LLM: {str(e)}")

    def __fineTuneLlm(self):
        if self.__trainingDataSet == "":
            Logging.GetInstance().Error("Training data set is missing. Please provide a valid data set using --train <data set>.")
            return False

        if self.__saveParamsFileName == "":
            Logging.GetInstance().Error("Parameters file name is missing. Please provide a valid file name using --saveparams <file_name>.")
            return False

        #try:
        with open(self.__trainingDataSet, "r") as file:
            inputText = file.read()

            trainRatio = 1
            batchSize = self.__config.getLearningConfig().getBatchSize()
            splitIdx = int(trainRatio * len(inputText))
            trainData = inputText[:splitIdx]

            dataSet = EncoderDecoderDataSet(config=self.__config, device=self.__device)
            dataSet.setRawText(trainData)
            #dataSet.tokenize()
            dataSet.buildInputTargetPair(maxLength = self.__config.getContextLength())

            gptDataLoader = EncoderDecoderDataLoader(dataSet, batchSize=batchSize, shuffle=True, numWorkers=10)
            dataLoader = gptDataLoader.getDataLoader()

            totalParameters = sum(p.numel() for p in self.getModel().parameters() if p.requires_grad)
            Logging.GetInstance().Info(f"Total trainable parameters: {totalParameters}, batch size: {batchSize}, number of batches per epoch: {len(dataLoader)}")

            self.getModel().setTokenizer(dataSet.getTokenizer())
            self.getModel().setDataLoader(gptDataLoader)
            self.getModel().setNumEpochs(self.__numEpochs)

            self.getModel.apply(lambda m: self.__initializeParameters(m))
            fineTuningModel = lora(self.__config, self.__model, self.__device)
            fineTuningModel.runTraining(saveFileName=self.__saveParamsFileName,
                                        loadFileName=self.__loadParamsFileName,
                                        isCheckpoint=True)
            # Add logic to fine-tune the LLM

    def run(self):
        if self.__llmRunType == LlmRunType.RUN:
            Logging.GetInstance().Info("Running LLM...")
            self.__runLlm()
            # Add logic to run the LLM
        elif self.__llmRunType == LlmRunType.TRAIN:
            Logging.GetInstance().Info(f"Training LLM with data set: {self.__trainingDataSet}")

            self.__trainLlm()
            # Add logic to train the LLM
        elif self.__llmRunType == LlmRunType.FINETUNE:
            Logging.GetInstance().Info(f"Fine-tuning LLM with data set: {self.__trainingDataSet}")
            self.__fineTuneLlm()
            # Add logic to fine-tune the LLM

        elif self.__llmRunType == LlmRunType.UNKNOWN:
            Logging.GetInstance().Error("LLM run type is unknown. Please specify --runllm or --train <data set>.")

if __name__ == "__main__":
    #try:
        configFileName = "../../Config/Test/TestConfig.json"  # Default config file
        torch.set_num_interop_threads(12)
        torch.set_num_threads(12)
        llmInstance = StartLlm(configFileName, device="cuda")
        if not llmInstance.getTerminateExecution():
            llmInstance.run()
    #except Exception as e:
    #    Logging.GetInstance().SetDetailedLogNeeded(False)
    #    Logging.GetInstance().Error(f"Error initializing LLM: {str(e)}")
    #    sys.exit(1)


