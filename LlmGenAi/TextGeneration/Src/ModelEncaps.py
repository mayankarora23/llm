#######################################################################
# File - ModelEncaps.py
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
from TransformerEncoder import TransformerEncoder
from LayerNormalization import LayerNormalization
from EncoderStack import EncoderStack
from DecoderStack import DecoderStack
from TokenEmbedding import EncoderDecoderDataSet
from TokenEmbedding import EncoderDecoderDataLoader
from LayerNormalization import LayerNormalization
from transformers import AutoTokenizer
from MyModelImplement import MyModel as llm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from torch.cuda.amp import autocast, GradScaler



import os


from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

class ModelEncaps():
    __config : Config
    __device : str
    __model : llm
    __isLearningRateEnhancement: bool = False
    __isCosineDecay: bool = False
    __enhancedLrScheduler = None
    __optimizer = None
    __dataLoader : EncoderDecoderDataLoader = None
    __tokenizer : AutoTokenizer = None
    __dataLoader : EncoderDecoderDataLoader = None
    __numEpochs : int = 1

    def __init__(self, config : Config, device: str = "cpu"):
        self.__config = config
        self.__device = device
        Logging.GetInstance().Info(f"ModelEncaps initializing on device: {self.__device}")
        self.__model = llm(config=self.__config, device=self.__device)
        self.__optimizer = self.__model.getOptimizer()
        Logging.GetInstance().Info("ModelEncaps initialized successfully")

    def setTokenizer(self, tokenizer: AutoTokenizer):
        self.__tokenizer = tokenizer
        if self.__device == "cuda":
            self.__model.module.setTokenizer(tokenizer)
        else:
            self.__model.setTokenizer(tokenizer)

    def setDataLoader(self, dataLoader: EncoderDecoderDataLoader):
        self.__dataLoader = dataLoader
        if self.__device == "cuda":
            self.__model.module.setDataLoader(dataLoader)
        else:
            self.__model.setDataLoader(dataLoader)

    def setNumEpochs(self, numEpochs: int):
        self.__numEpochs = numEpochs
        if self.__device == "cuda":
            self.__model.module.setNumEpochs(numEpochs)
        else:
            self.__model.setNumEpochs(numEpochs)

    def initDevices(self):
        if self.__device == "cuda":
            if self.__config.getLlmStartConfig().useDdp():
                self.initializeDistributed()
            else:
                numGPUs = torch.cuda.device_count()
                Logging.GetInstance().Info(f"Number of GPUs available: {numGPUs}")
                gpuArrays = [i for i in range(numGPUs)]
                Logging.GetInstance().Info(f"Using GPUs: {gpuArrays}")
                self.__model = nn.DataParallel(self.__model, device_ids=gpuArrays)
                self.__model = self.__model.cuda()
        else:
            self.__model = self.__model.to(torch.device(self.__device))

    def __getEnhancedLearningRateScheduler(self,
                                   warmupScheduler,
                                   cosineDecaySteps,
                                   warmupSteps):
        combinedScheduler = None
        if self.__isCosineDecay:
            cosineDecayScheduler = CosineAnnealingLR(
                self.__model.module.getOptimizer(),
                T_max=cosineDecaySteps,
                eta_min=self.__config.getLearningConfig().getLearningRate() * 0.1 # 10% of initial learning rate
            )

            combinedScheduler = SequentialLR(
                self.__model.getOptimizer(),
                schedulers=[warmupScheduler, cosineDecayScheduler],
                milestones=[warmupSteps]
            )
        else:
            combinedScheduler = warmupScheduler

        return combinedScheduler

    def __initEnhancedLearningRateScheduler(self):
        if self.__isLearningRateEnhancement:
            # Initialize warmup and cosine decay Steps            
            dataLoadLength = len(self.__dataLoader.getDataLoader())
            totalSteps = dataLoadLength * self.__numEpochs
            warmupSteps = int(dataLoadLength * self.__config.getLearningConfig().getWarmupStepPercent())
            Logging.GetInstance().Debug(f"Warmup steps: {warmupSteps}")
            cosineDecaySteps = totalSteps - warmupSteps
            Logging.GetInstance().Debug(f"Cosine decay steps: {cosineDecaySteps}")

            # Initialize schedulers
            warmupScheduler = LambdaLR(
                self.__model.module.getOptimizer(),
                lr_lambda=lambda step: min((step + 1) / warmupSteps, 1.0))

            self.__enhancedLrScheduler = self.__getEnhancedLearningRateScheduler(warmupScheduler,
                                                                                 cosineDecaySteps,
                                                                                 warmupSteps)

    def initEnhancedLearningRate(self):
        self.__isLearningRateEnhancement = self.__config.getLearningConfig().isLrWarmupEnabled()
        self.__isCosineDecay = self.__config.getLearningConfig().isCosineDecayEnabled()
        self.__initEnhancedLearningRateScheduler()

    def __batchLoop(self, epoch: int, isCheckpoint: bool, saveFileName: str):
        if self.__device == "cuda":
            device = next(self.__model.module.parameters()).device
        else:
            device = next(self.__model.parameters()).device
        batchIndex = 0
        dataLoadLength = len(self.__dataLoader.getDataLoader())
        gradientAccumulationSteps = self.__config.getGradientAccumulationStepSize()

        if self.__device == "cuda":
            self.__scaler = torch.amp.GradScaler(enabled=(self.__device == "cuda"))
            self._GradScaler_initialized = True

        if self.__device == "cuda":
            self.__model.module.getOptimizer().zero_grad(set_to_none=True)
        else:
            self.__model.getOptimizer().zero_grad(set_to_none=True)

        for inputIds, targetIds in self.__dataLoader.getDataLoader():
            inputIds = inputIds.to(device)
            targetIds = targetIds.to(device)

            # Shift targetIds to get decoder_input_ids
            decoderInputIds = torch.cat(
                [torch.full((targetIds.size(0), 1), self.__tokenizer.pad_token_id, dtype=torch.long, device=device),
                targetIds[:, :-1]],
                dim=1
            )

            expectedOutputIds = targetIds
            learningRateUsed = 0

            isDdp = self.__device == "cuda" and hasattr(self.__model, "module")
            isSyncNeeded = (batchIndex % gradientAccumulationSteps != 0)

            # Don't synchronize gradients every step if using gradient accumulation
            if isDdp and isSyncNeeded:
                ddpContext = self.__model.no_sync()
            else:
                # nullcontext from contextlib is a no-op context manager
                from contextlib import nullcontext
                ddpContext = nullcontext()

            with ddpContext:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = self.__model(inputIds, decoderInputIds)

                    if self.__device == "cuda":
                        loss = self.__model.module.getLoss(logits, expectedOutputIds)
                    else:
                        loss = self.__model.getLoss(logits, expectedOutputIds)

                    loss = loss / gradientAccumulationSteps

                    entropy = 0.0
                    if epoch > 0 or batchIndex > 0:
                        if self.__device == "cuda":
                            entropy = self.__model.module.getDecoderStack().getEntropy(epoch, batchIndex)
                        else:
                            entropy = self.__model.getDecoderStack().getEntropy(epoch, batchIndex)

                    # Below section of code is to add entropy regularization to the
                    # loss function. This is done to encourage the model to have
                    # lower entropy in its cross attention weights
                    if self.__config.getLearningConfig().getCrossAttRegRate() is not None:
                        isEntropyRegularizationEnabled = True
                    else:
                        isEntropyRegularizationEnabled = False

                    if isEntropyRegularizationEnabled:
                        loss = loss + self.__config.getLearningConfig().getCrossAttRegRate() * entropy  # Adding entropy as a regularization term

                # End of autocast
            # End of ddpContext
            avgConf = 0.0
            with torch.no_grad():
                logitsDitached = logits.detach()
                probs = torch.nn.functional.softmax(logitsDitached, dim=-1)
                tokenConfidence = probs.gather(dim=2, index=expectedOutputIds.unsqueeze(-1)).squeeze(-1)
                avgConf = tokenConfidence.mean().item()

            # Guard against NaN loss
            if not torch.isfinite(loss):
                Logging.GetInstance().Warn(f"Skipping batch {batchIndex} due to non-finite loss")
                if self.__device == "cuda":
                    self.__model.module.getOptimizer().zero_grad(set_to_none=True)
                else:
                    self.__model.getOptimizer().zero_grad(set_to_none=True)

                batchIndex += 1
                continue

            if self.__device == "cuda":
                perplexity = self.__model.module.getPerplexity(loss)
            else:
                perplexity = self.__model.getPerplexity(loss)

            if self.__device == "cuda":
                # Guard against NaN loss
                self.__scaler.scale(loss).backward()
            else:
                loss.backward()

            Logging.GetInstance().Debug(f"Gradient Clipping - "
                                       f"{self.__config.getLearningConfig().getGradientClipping()}")

            if (batchIndex + 1) % gradientAccumulationSteps == 0:
                if self.__device == "cuda":
                    # Unscale gradients before clipping
                    self.__scaler.unscale_(self.__model.module.getOptimizer())

                # clipping the gradients to avoid exploding gradients
                # exploding gradients means the gradients become too large
                # and cause the model to diverge, to start with we are using
                # max_norm of 1.0, which is conservative and should work
                if self.__device == "cuda":
                    torch.nn.utils.clip_grad_norm_(
                        self.__model.module.parameters(),
                        max_norm=self.__config.getLearningConfig().getGradientClipping())
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.__model.parameters(),
                        max_norm=self.__config.getLearningConfig().getGradientClipping())

                if self.__device == "cuda":
                    self.__scaler.step(self.__model.module.getOptimizer())
                    self.__scaler.update()
                else:
                    self.__model.getOptimizer().step()

                if self.__isLearningRateEnhancement:
                    # Step the scheduler if using learning rate enhancement
                    self.__enhancedLrScheduler.step()
                    learningRateUsed = self.__enhancedLrScheduler.get_last_lr()[0]
                else:
                    learningRateUsed = self.__model.module.getOptimizer().param_groups[0]['lr']

                if self.__device == "cuda":
                    self.__model.module.getOptimizer().zero_grad(set_to_none=True)
                else:
                    self.__model.getOptimizer().zero_grad(set_to_none=True)

            if batchIndex % 100 == 0 and isCheckpoint and saveFileName and dist.get_rank() == 0:
                if self.__device == "cuda":
                    self.__model.module.checkpoint(saveFileName,
                                    epoch,
                                    self.__isLearningRateEnhancement,
                                    self.__isCosineDecay,
                                    self.__enhancedLrScheduler)
                else:
                    self.__model.checkpoint(saveFileName,
                                    epoch,
                                    self.__isLearningRateEnhancement,
                                    self.__isCosineDecay,
                                    self.__enhancedLrScheduler)
                Logging.GetInstance().Info(f"Model checkpoint saved to {saveFileName}")

            Logging.GetInstance().Info(f"{batchIndex}/{dataLoadLength} [{epoch+1}/{self.__numEpochs}] "
                                       f"Loss: {loss:.2f} Perplexity: {perplexity:.2f}, "
                                       f"LR: {learningRateUsed:.6f}, "
                                       f"tokenConfidence: {avgConf:.4f}, "
                                       f"Cross-attn Entropy: {entropy:.4f}")
            batchIndex += 1

    def trainingLoop(self, isCheckpoint: bool, saveFileName: str):
        dataLoadLength = len(self.__dataLoader.getDataLoader())
        for epoch in range(self.__numEpochs):
            self.__batchLoop(epoch, isCheckpoint, saveFileName)

            if isCheckpoint and saveFileName and dist.get_rank() == 0:
                if self.__device == "cuda":
                    self.__model.module.checkpoint(saveFileName,
                                epoch,
                                self.__isLearningRateEnhancement,
                                self.__isCosineDecay,
                                self.__enhancedLrScheduler)
                else:
                    self.__model.checkpoint(saveFileName,
                                    epoch,
                                    self.__isLearningRateEnhancement,
                                    self.__isCosineDecay,
                                    self.__enhancedLrScheduler)
                Logging.GetInstance().Info(f"Model checkpoint saved to {saveFileName}")
            Logging.GetInstance().Info(f"Epoch {epoch + 1}/{self.__numEpochs} completed")

    def runTraining(self, saveFileName: str = "",
                    loadFileName: str = "",
                    isCheckpoint: bool = False,
                    isFineTuning: bool = False):

        dataLoadLength = len(self.__dataLoader.getDataLoader())

        if loadFileName:
            #loadFileHandler = open(loadFileName, "rb")
            #self.load_state_dict(torch.load(loadFileHandler))
            epoch, self.__enhancedLrScheduler, self.__isLearningRateEnhancement, self.__isCosineDecay =\
                self.loadCheckpoint(loadFileName,
                                    dataLoadLength,
                                    isFineTuning=isFineTuning)
            Logging.GetInstance().Info(f"Model loaded from {loadFileName}")

        self.trainingLoop(isCheckpoint, saveFileName)

    def getModel(self) -> llm:
        return self.__model

    def generateText(self, startingInput: torch.Tensor, maxNewToken: int) -> torch.Tensor:
        if self.__device == "cuda":
            return self.__model.module.generateText(startingInput, maxNewToken)
        else:
            return self.__model.generateText(startingInput, maxNewToken)

    def setupDistributed(self, localRank: int):
        dist.init_process_group("nccl")
        localRank = int(os.environ["LOCAL_RANK"]) # GPU index on this node
        torch.cuda.set_device(localRank)

        Logging.GetInstance().Info(f"Distributed training setup on local rank: {localRank}")
        return localRank

    def initializeDistributed(self):
        localRank = self.setupDistributed(localRank=0)
        device = torch.device(f"cuda:{localRank}")

        self.__model = self.__model.to(device)
        self.__model = DDP(self.__model, device_ids=[localRank])

        Logging.GetInstance().Info("Model wrapped with DistributedDataParallel for distributed training")
