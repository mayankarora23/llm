#######################################################################
# File - Config.py
# Author - Mayank Arora
# Description - This file contains the implementation of extracting and
#               holding the configuration for the transformers.
#######################################################################

import sys

sys.path.append("../../../Common")
from Logging import Logging
from Logging import LogLevel

import json

class LearningConfig:
    __learningRate: float
    __beta1: float
    __beta2: float
    __epsilon: float
    __weightDecay: float
    __warmupStepPercent: int
    __repetitionPenalty: float
    __labelSmoothing: float
    __decoderDropoutScaling: float
    __crossAttRegRate: float
    __encoderLayerDropRate: float
    __decoderLayerDropRate: float
    __denoisingMaskingRate: float
    __batchSize: int
    __gradientClipping: float
    __enableCosineDecay: bool
    __enableLrWarmup: bool
    def __init__(self):
        self.__learningRate = 0.0001
        self.__beta1 = None
        self.__beta2 = None
        self.__epsilon = None
        self.__weightDecay = 0.1
        self.__warmupStepPercent = 0.1  # 10% of total steps
        self.__repetitionPenalty = 1.2  # Default value for repetition penalty
        self.__labelSmoothing = 0.1  # Default value for label smoothing
        self.__decoderDropoutScaling = 1.0  # Default dropout scaling for decoder
        self.__crossAttRegRate = None # Default value for cross attention regularization
        self.__encoderLayerDropRate = 0.0 # Default value for encoder layer drop rate
        self.__decoderLayerDropRate = 0.0 # Default value for decoder layer drop rate
        self.__denoisingMaskingRate = 0.3 # Default value for denoising masking rate
        self.__batchSize = 2 # Default value for batch size
        self.__gradientClipping = 1.0
        self.__enableCosineDecay = False # Default value for cosine decay
        self.__enableLrWarmup = False # Default value for learning rate warmup

    def loadLearningConfig(self, configData: dict):
        try:
            learningConfig = configData.get("learningConfig")
            if not learningConfig:
                raise ValueError("learningConfig not found in the configuration.")

            self.__learningRate = learningConfig.get("learningRate")
            if not self.__learningRate:
                raise ValueError("learningRate not found in the configuration.")
            self.__beta1 = learningConfig.get("beta1")
            if not self.__beta1:
                self.__beta1 = None
            self.__beta2 = learningConfig.get("beta2")
            if not self.__beta2:
                self.__beta2 = None
            self.__epsilon = learningConfig.get("epsilon")
            if not self.__epsilon:
                self.__epsilon = None
            self.__weightDecay = learningConfig.get("weightDecay")
            if not self.__weightDecay:
                raise ValueError("weightDecay not found in the configuration.")
            self.__warmupStepPercent = learningConfig.get("warmupStepPercent")
            if not self.__warmupStepPercent:
                self.__warmupStepPercent = None
            self.__repetitionPenalty = learningConfig.get("repetitionPenalty")
            if not self.__repetitionPenalty:
                self.__repetitionPenalty = 1.2
            self.__labelSmoothing = learningConfig.get("labelSmoothing")
            if not self.__labelSmoothing:
                self.__labelSmoothing = 0.1
            self.__decoderDropoutScaling = learningConfig.get("decoderDropoutScaling")
            if not self.__decoderDropoutScaling:
                self.__decoderDropoutScaling = 1.0
            self.__crossAttRegRate = learningConfig.get("crossAttRegRate")
            self.__encoderLayerDropRate = learningConfig.get("encoderLayerDropRate")
            if not self.__encoderLayerDropRate:
                self.__encoderLayerDropRate = 0.0
            self.__decoderLayerDropRate = learningConfig.get("decoderLayerDropRate")
            if not self.__decoderLayerDropRate:
                self.__decoderLayerDropRate = 0.0
            self.__denoisingMaskingRate = learningConfig.get("denoisingMaskingRate")
            if not self.__denoisingMaskingRate:
                self.__denoisingMaskingRate = 0.3
            self.__batchSize = learningConfig.get("batchSize")
            if not self.__batchSize:
                self.__batchSize = 2
            self.__gradientClipping = learningConfig.get("gradientClipping")
            if not self.__gradientClipping:
                self.__gradientClipping = 1.0
            self.__enableCosineDecay = learningConfig.get("enableCosineDecay")
            if self.__enableCosineDecay is None:
                self.__enableCosineDecay = False
            self.__enableLrWarmup = learningConfig.get("enableLrWarmup")
            if self.__enableLrWarmup is None:
                self.__enableLrWarmup = False

            Logging.GetInstance().Debug(f"Learning Config loaded:\n"
                                        f"  learningRate: {self.__learningRate}\n"
                                        f"  beta1: {self.__beta1}\n"
                                        f"  beta2: {self.__beta2}\n"
                                        f"  epsilon: {self.__epsilon}\n"
                                        f"  weightDecay: {self.__weightDecay}\n"
                                        f"  warmupStepPercent: {self.__warmupStepPercent}\n"
                                        f"  repetitionPenalty: {self.__repetitionPenalty}\n"
                                        f"  labelSmoothing: {self.__labelSmoothing}")

        except Exception as e:
            Logging.GetInstance().Error(f"Error loading learning config: {e}")

    def getLearningRate(self) -> float:
        return self.__learningRate

    def getBeta1(self) -> float:
        return self.__beta1

    def getBeta2(self) -> float:
        return self.__beta2

    def getEpsilon(self) -> float:
        return self.__epsilon

    def getWeightDecay(self) -> float:
        return self.__weightDecay

    def getWarmupStepPercent(self) -> int:
        return self.__warmupStepPercent

    def getRepetitionPenalty(self) -> float:
        return self.__repetitionPenalty

    def getLabelSmoothing(self) -> float:
        return self.__labelSmoothing

    def getDecoderDropoutScaling(self) -> float:
        return self.__decoderDropoutScaling

    def getCrossAttRegRate(self):
        return self.__crossAttRegRate

    def getEncoderLayerDropRate(self) -> float:
        return self.__encoderLayerDropRate

    def getDecoderLayerDropRate(self) -> float:
        return self.__decoderLayerDropRate

    def getDenoisingMaskingRate(self) -> float:
        return self.__denoisingMaskingRate

    def getBatchSize(self) -> int:
        return self.__batchSize

    def getGradientClipping(self) -> float:
        return self.__gradientClipping

    def isCosineDecayEnabled(self) -> bool:
        return self.__enableCosineDecay

    def isLrWarmupEnabled(self) -> bool:
        return self.__enableLrWarmup

class AttentionConfig:
    __multiHeadAttentionDropoutRate: float
    __crossMultiHeadAttentionDropoutRate: float

    def __init__(self):
        self.__multiHeadAttentionDropoutRate = 0.1
        self.__crossMultiHeadAttentionDropoutRate = 0.1
    def loadAttentionConfig(self, configData: dict):
        try:
            attentionConfig = configData.get("attentionConfig")
            if not attentionConfig:
                raise ValueError("attentionConfig not found in the configuration.")

            self.__multiHeadAttentionDropoutRate =\
                attentionConfig.get("multiHeadAttentionDropoutRate")
            self.__crossMultiHeadAttentionDropoutRate =\
                attentionConfig.get("crossHeadAttentionDropoutRate")

            Logging.GetInstance().Debug(f"Attention Config loaded:\n"
                                        f"  multiHeadAttentionDropoutRate: {self.__multiHeadAttentionDropoutRate}\n"
                                        f"  crossMultiHeadAttentionDropoutRate: {self.__crossMultiHeadAttentionDropoutRate}")

        except Exception as e:
            Logging.GetInstance().Error(f"Error loading attention config: {e}")

    def getMultiHeadAttentionDropoutRate(self) -> float:
        return self.__multiHeadAttentionDropoutRate

    def getCrossMultiHeadAttentionDropoutRate(self) -> float:
        return self.__crossMultiHeadAttentionDropoutRate

class TransformerEncoderConfig:
    __postMultiHeadAttentionDropoutRate: float
    _postFeedForwardDropoutRate: float

    def __init__(self):
        self.__postMultiHeadAttentionDropoutRate = 0.1
        self.__postFeedForwardDropoutRate = 0.1

    def loadTransformerEncoderConfig(self, configData: dict):
        try:
            transformerEncoderConfig = configData.get("transformerEncoderConfig")
            if not transformerEncoderConfig:
                raise ValueError("transformerEncoderConfig not found in the configuration.")

            self.__postMultiHeadAttentionDropoutRate =\
                transformerEncoderConfig.get("postMultiHeadAttentionDropoutRate")
            self.__postFeedForwardDropoutRate =\
                transformerEncoderConfig.get("postFeedForwardDropoutRate")

            Logging.GetInstance().Debug(f"Transformer Encoder Config loaded:\n"
                                        f"  postMultiHeadAttentionDropoutRate: {self.__postMultiHeadAttentionDropoutRate}\n"
                                        f"  postFeedForwardDropoutRate: {self.__postFeedForwardDropoutRate}")

        except Exception as e:
            Logging.GetInstance().Error(f"Error loading transformer encoder config: {e}")

    def getPostMultiHeadAttentionDropoutRate(self) -> float:
        return self.__postMultiHeadAttentionDropoutRate

    def getPostFeedForwardDropoutRate(self) -> float:
        return self.__postFeedForwardDropoutRate

class GeneratorConfig:
    __temperature: float
    __topK: int

    def __init__(self):
        self.__temperature = 1.0
        self.__topK = 40

    def loadGeneratorConfig(self, configData: dict):
        try:
            generatorConfig = configData.get("generatorConfig")
            if not generatorConfig:
                raise ValueError("generatorConfig not found in the configuration.")
            self.__temperature = generatorConfig.get("temperature")
            if not self.__temperature:
                self.__temperature = 0.7
            self.__topK = generatorConfig.get("topK")
            if not self.__topK:
                self.__topK = 40
            Logging.GetInstance().Debug(f"Generator Config loaded:\n"
                                        f"  temperature: {self.__temperature}\n"
                                        f"  topK: {self.__topK}")
        except Exception as e:
            Logging.GetInstance().Error(f"Error loading generator config: {e}")

    def getTemperature(self) -> float:
        return self.__temperature

    def getTopK(self) -> int:
        return self.__topK

class DataSetConfig:
    __startOfFixedData: int
    __endOfFixedData: int
    __totalDataList: int
    __useVariableDataSet: bool

    def __init__(self):
        self.__startOfFixedData = 0
        self.__endOfFixedData = 30000
        self.__totalDataList = 30000
        self.__useVariableDataSet = False

    def loadDataSetConfig(self, configData: dict):
        try:
            dataSetConfig = configData.get("dataSetConfig")
            if not dataSetConfig:
                raise ValueError("dataSetConfig not found in the configuration.")
            self.__startOfFixedData = dataSetConfig.get("startOfFixedData")
            if self.__startOfFixedData is None:
                self.__startOfFixedData = 0
            self.__endOfFixedData = dataSetConfig.get("endOfFixedData")
            if self.__endOfFixedData is None:
                self.__endOfFixedData = 30000
            self.__totalDataList = dataSetConfig.get("totalDataList")
            if self.__totalDataList is None:
                self.__totalDataList = 30000
            self.__useVariableDataSet = dataSetConfig.get("useVariableDataSet")
            if self.__useVariableDataSet is None:
                self.__useVariableDataSet = False
            Logging.GetInstance().Debug(f"DataSet Config loaded:\n"
                                        f"  startOfFixedData: {self.__startOfFixedData}\n"
                                        f"  endOfFixedData: {self.__endOfFixedData}\n"
                                        f"  useVariableDataSet: {self.__useVariableDataSet}")
        except Exception as e:
            Logging.GetInstance().Error(f"Error loading dataset config: {e}")

    def getStartOfFixedData(self) -> int:
        return self.__startOfFixedData

    def getEndOfFixedData(self) -> int:
        return self.__endOfFixedData

    def getTotalDataList(self) -> int:
        return self.__totalDataList

    def useVariableDataSet(self) -> bool:
        return self.__useVariableDataSet

class CheckpointConfig:
    __useOptimiserStateFromCheckpoint: bool
    __useLrWarmupStateFromCheckpoint: bool
    __useCossineDecayStateFromCheckpoint: bool
    __overRideOptimiserLrFromConfig: bool

    def __init__(self):
        self.__useOptimiserStateFromCheckpoint = False
        self.__useLrWarmupStateFromCheckpoint = False
        self.__useCossineDecayStateFromCheckpoint = False
        self.__overRideOptimiserLrFromConfig = False

    def loadCheckpointConfig(self, configData: dict):
        try:
            checkpointConfig = configData.get("checkpointConfig")
            if not checkpointConfig:
                raise ValueError("checkpointConfig not found in the configuration.")
            self.__useOptimiserStateFromCheckpoint = checkpointConfig.get("useOptimiserStateFromCheckpoint")
            if self.__useOptimiserStateFromCheckpoint is None:
                self.__useOptimiserStateFromCheckpoint = False
            self.__useLrWarmupStateFromCheckpoint = checkpointConfig.get("useLrWarmupStateFromCheckpoint")
            if self.__useLrWarmupStateFromCheckpoint is None:
                self.__useLrWarmupStateFromCheckpoint = False
            self.__useCossineDecayStateFromCheckpoint = checkpointConfig.get("useCossineDecayStateFromCheckpoint")
            if self.__useCossineDecayStateFromCheckpoint is None:
                self.__useCossineDecayStateFromCheckpoint = False
            self.__overRideOptimiserLrFromConfig = checkpointConfig.get("overRideOptimiserLrFromConfig")
            if self.__overRideOptimiserLrFromConfig is None:
                self.__overRideOptimiserLrFromConfig = False
            Logging.GetInstance().Debug(f"Checkpoint Config loaded:\n"
                                        f"  useOptimiserStateFromCheckpoint: {self.__useOptimiserStateFromCheckpoint}\n"
                                        f"  useLrWarmupStateFromCheckpoint: {self.__useLrWarmupStateFromCheckpoint}\n"
                                        f"  useCossineDecayStateFromCheckpoint: {self.__useCossineDecayStateFromCheckpoint}\n"
                                        f"  overRideOptimiserLrFromConfig: {self.__overRideOptimiserLrFromConfig}")
        except Exception as e:
            Logging.GetInstance().Error(f"Error loading checkpoint config: {e}")

    def useOptimiserStateFromCheckpoint(self) -> bool:
        return self.__useOptimiserStateFromCheckpoint

    def useLrWarmupStateFromCheckpoint(self) -> bool:
        return self.__useLrWarmupStateFromCheckpoint

    def useCossineDecayStateFromCheckpoint(self) -> bool:
        return self.__useCossineDecayStateFromCheckpoint

    def overRideOptimiserLrFromConfig(self) -> bool:
        return self.__overRideOptimiserLrFromConfig

class LlmStartConfig:
    __device: str
    __numEpochs: int
    __isTraining: bool
    __runLlm: bool
    __isFineTuning: bool
    __saveParamsFileName: str
    __useLoadParams: bool
    __loadParamsFileName: str
    __dataSetPath: str
    __useDdp: bool

    def __init__(self):
        self.__device = "cpu"
        self.__numEpochs = 1
        self.__isTraining = True
        self.__runLlm = True
        self.__isFineTuning = False
        self.__saveParamsFileName = ""
        self.__useLoadParams = False
        self.__loadParamsFileName = ""
        self.__dataSetPath = ""
        self.__useDdp = False

    def loadLlmStartConfig(self, configData: dict):
        try:
            llmStartconfig = configData.get("llmStartConfig")
            if not llmStartconfig:
                raise ValueError("llmStartConfig not found in the configuration.")
            self.__device = llmStartconfig.get("device")
            if not self.__device:
                self.__device = "cpu"
            self.__numEpochs = llmStartconfig.get("numEpochs")
            if not self.__numEpochs:
                self.__numEpochs = 1
            self.__isTraining = llmStartconfig.get("isTraining")
            if self.__isTraining is None:
                self.__isTraining = False
            self.__runLlm = llmStartconfig.get("runLlm")
            if self.__runLlm is None:
                self.__runLlm = False
            self.__isFineTuning = llmStartconfig.get("isFineTuning")
            if self.__isFineTuning is None:
                self.__isFineTuning = False
            self.__saveParamsFileName = llmStartconfig.get("saveParamsFileName")
            if not self.__saveParamsFileName:
                self.__saveParamsFileName = ""
            self.__useLoadParams = llmStartconfig.get("useLoadParams")
            if self.__useLoadParams is None:
                self.__useLoadParams = False
            self.__loadParamsFileName = llmStartconfig.get("loadParamsFileName")
            if not self.__loadParamsFileName:
                self.__loadParamsFileName = ""
            self.__dataSetPath = llmStartconfig.get("dataSetPath")
            if not self.__dataSetPath:
                self.__dataSetPath = ""
            self.__useDdp = llmStartconfig.get("useDdp")
            if self.__useDdp is None:
                self.__useDdp = False
            Logging.GetInstance().Debug(f"Llm Start Config loaded:\n"
                                        f"  device: {self.__device}\n"
                                        f"  numEpochs: {self.__numEpochs}\n"
                                        f"  isTraining: {self.__isTraining}\n"
                                        f"  runLlm: {self.__runLlm}\n"
                                        f"  isFineTuning: {self.__isFineTuning}\n"
                                        f"  saveParamsFileName: {self.__saveParamsFileName}\n"
                                        f"  useLoadParams: {self.__useLoadParams}\n"
                                        f"  loadParamsFileName: {self.__loadParamsFileName}\n"
                                        f"  dataSetPath: {self.__dataSetPath}\n"
                                        f"  useDdp: {self.__useDdp}")
        except Exception as e:
            Logging.GetInstance().Error(f"Error loading checkpoint config: {e}")

    def getDevice(self) -> str:
        return self.__device

    def getNumEpochs(self) -> int:
        return self.__numEpochs

    def isTraining(self) -> bool:
        return self.__isTraining

    def runLlm(self) -> bool:
        return self.__runLlm

    def isFineTuning(self) -> bool:
        return self.__isFineTuning

    def getSaveParamsFileName(self) -> str:
        return self.__saveParamsFileName

    def useLoadParams(self) -> bool:
        return self.__useLoadParams 

    def getLoadParamsFileName(self) -> str:
        return self.__loadParamsFileName

    def getDataSetPath(self) -> str:
        return self.__dataSetPath

    def useDdp(self) -> bool:
        return self.__useDdp

class Config:
    __vocabSize: int
    __contextLength: int
    __embeddingDimension: int
    __attentionHeads: int
    __numLayers: int
    __dropoutRate: float
    __useQueryKeyValueBias: bool
    __usePreNormalisation: bool
    __gradientAccumulationStepSize: int
    __configFilePath: str
    __configType: str
    __learningConfig: LearningConfig
    __attentionConfig: AttentionConfig
    __transformerEncoderConfig: TransformerEncoderConfig
    __generatorConfig: GeneratorConfig
    __dataSetConfig: DataSetConfig
    __checkpointConfig: CheckpointConfig
    __llmStartConfig: LlmStartConfig

    def __init__(self, configFilePath: str = "./Config.json"):
        Logging.GetInstance().Debug(f"Initializing Config with file: {configFilePath}")
        self.__configFilePath = configFilePath

        self.__vocabSize = 50257  # Default value for GPT-2
        self.__contextLength = 1024  # Default value for GPT-2
        self.__embeddingDimension = 768  # Default value for GPT-2
        self.__attentionHeads = 12  # Default value for GPT-2
        self.__numLayers = 12  # Default value for GPT-2
        self.__dropoutRate = 0.1  # Default value for GPT-2
        self.__useQueryKeyValueBias = False  # Default value for GPT-2
        self.__usePreNormalisation = True # Default value for Pre-Normalization
        self.__gradientAccumulationStepSize = 1 # Default value for gradient accumulation step size
        self.__configType = ""
        self.__learningConfig = LearningConfig()
        self.__attentionConfig = AttentionConfig()
        self.__transformerEncoderConfig = TransformerEncoderConfig()
        self.__generatorConfig = GeneratorConfig()
        self.__dataSetConfig = DataSetConfig()
        self.__checkpointConfig = CheckpointConfig()
        self.__llmStartConfig = LlmStartConfig()


    def loadConfig(self, configType: str = ""):
        try:
            with open(self.__configFilePath, 'r') as configHandler:
                configData = json.load(configHandler)
                if configType == "":
                    configType = configData.get("useGptConfig")
                gptConfig = configData.get(configType)
                if not gptConfig:
                    raise ValueError(f"Configuration type '{configType}' not found in the config file.")

                self.__configType = configType

                # Load GPT configuration parameters
                self.__vocabSize = gptConfig.get("vocabularySize")
                if not self.__vocabSize:
                    raise ValueError("vocabularySize not found in the configuration.")
                self.__contextLength = gptConfig.get("contextLength")
                if not self.__contextLength:
                    raise ValueError("contextLength not found in the configuration.")
                self.__embeddingDimension = gptConfig.get("embeddingDimension")
                if not self.__embeddingDimension:
                    raise ValueError("embeddingDimension not found in the configuration.")
                self.__attentionHeads = gptConfig.get("attentionHeads")
                if not self.__attentionHeads:
                    raise ValueError("attentionHeads not found in the configuration.")
                self.__numLayers = gptConfig.get("numLayers")
                if not self.__numLayers:
                    raise ValueError("numLayers not found in the configuration.")
                self.__dropoutRate = gptConfig.get("dropoutRate")
                if self.__dropoutRate is None:
                    raise ValueError("dropoutRate not found in the configuration.")
                self.__useQueryKeyValueBias = gptConfig.get("useQueryKeyValueBias")
                if self.__useQueryKeyValueBias is None:
                    raise ValueError("useQueryKeyValueBias not found in the configuration.")
                self.__usePreNormalisation = gptConfig.get("usePreNormalisation")
                if self.__usePreNormalisation is None:
                    self.__usePreNormalisation = True
                self.__gradientAccumulationStepSize = gptConfig.get("gradientAccumStepSize")
                if not self.__gradientAccumulationStepSize:
                    self.__gradientAccumulationStepSize = 1

                Logging.GetInstance().Debug(f"{configType} Config loaded:\n"
                                            f"  vocabSize: {self.__vocabSize}\n"
                                            f"  contextLength: {self.__contextLength}\n"
                                            f"  embeddingDimension: {self.__embeddingDimension}\n"
                                            f"  attentionHeads: {self.__attentionHeads}\n"
                                            f"  numLayers: {self.__numLayers}\n"
                                            f"  dropoutRate: {self.__dropoutRate}\n"
                                            f"  useQueryKeyValueBias: {self.__useQueryKeyValueBias}")

                # Load learning rate configuration parameters
                self.__learningConfig.loadLearningConfig(configData)

                # Load attention configuration parameters
                self.__attentionConfig.loadAttentionConfig(configData)

                # Load transformer encoder configuration parameters
                self.__transformerEncoderConfig.loadTransformerEncoderConfig(configData)

                # Load generator configuration parameters
                self.__generatorConfig.loadGeneratorConfig(configData)

                # Load dataset configuration parameters
                self.__dataSetConfig.loadDataSetConfig(configData)

                # Load checkpoint configuration parameters
                self.__checkpointConfig.loadCheckpointConfig(configData)

                # Load llm start configuration parameters
                self.__llmStartConfig.loadLlmStartConfig(configData)

        except FileNotFoundError:
            Logging.GetInstance().Error(f"Config file not found: {self.__configFilePath}")
        except json.JSONDecodeError:
            Logging.GetInstance().Error(f"Error decoding JSON from config file: {self.__configFilePath}")
        except Exception as e:
            Logging.GetInstance().Error(f"Unexpected error while loading config: {e}")
            raise e

    def getVocabSize(self) -> int:
        return self.__vocabSize

    def getContextLength(self) -> int:
        return self.__contextLength

    def getEmbeddingDimension(self) -> int:
        return self.__embeddingDimension

    def getAttentionHeads(self) -> int:
        return self.__attentionHeads

    def getNumLayers(self) -> int:
        return self.__numLayers

    def getDropoutRate(self) -> float:
        return self.__dropoutRate

    def useQueryKeyValueBias(self) -> bool:
        return self.__useQueryKeyValueBias

    def usePreNormalisation(self) -> bool:
        return self.__usePreNormalisation

    def getGradientAccumulationStepSize(self) -> int:
        return self.__gradientAccumulationStepSize

    def getConfigType(self) -> str:
        return self.__configType

    def getLearningConfig(self) -> LearningConfig:
        return self.__learningConfig

    def getAttentionConfig(self) -> AttentionConfig:
        return self.__attentionConfig

    def getTransformerEncoderConfig(self) -> TransformerEncoderConfig:
        return self.__transformerEncoderConfig

    def getGeneratorConfig(self) -> GeneratorConfig:
        return self.__generatorConfig

    def getDataSetConfig(self) -> DataSetConfig:
        return self.__dataSetConfig

    def getCheckpointConfig(self) -> CheckpointConfig:
        return self.__checkpointConfig

    def getLlmStartConfig(self) -> LlmStartConfig:
        return self.__llmStartConfig