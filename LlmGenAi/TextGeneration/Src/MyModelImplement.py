#######################################################################
# File - MyModelImplement.py
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

class MyModel(nn.Module):
    __config: Config
    __encoderStack: EncoderStack
    __decoderStack: DecoderStack
    __outputLayer: nn.Linear
    __dataLoader: EncoderDecoderDataLoader
    __validationDataLoader: EncoderDecoderDataLoader
    __optimizer: torch.optim.Optimizer
    __numEpochs: int = 4
    __device: str
    __repetitionPenalty: float = 5
    __generatedTokens = [[]]
    __tokenizer: AutoTokenizer
    __dropout: nn.Dropout

    def __initEncoderDecoderStacks(self):
        # Initialize encoder stack
        self.__encoderStack = EncoderStack(self.__config)
        Logging.GetInstance().Debug(f"Encoder stack initialized with config: {self.__config.getConfigType()}")

        # Initialize decoder stack
        self.__decoderStack = DecoderStack(self.__config)
        Logging.GetInstance().Debug(f"Decoder stack initialized with config: {self.__config.getConfigType()}")

    def __init__(self,
                 configFileName="",
                 config: Config = None,
                 device: str = "cpu",
                 numEpochs: int = 4):
        super().__init__()

        if configFileName != "":
            self.__config = Config(configFileName)
            self.__config.loadConfig()

        elif config is not None:
            self.__config = config

        self.__numEpochs = numEpochs

        self.__initEncoderDecoderStacks()

        # Initialize output layer
        self.__outputLayer = nn.Linear(
            self.__config.getEmbeddingDimension(),
            self.__config.getVocabSize(),
            bias=False
        )

        self.__dataLoader = None
        self.__validationDataLoader = None

        learningRate, betas, eps, weightDecay = self.__getOptimizerParamsFromConfig()

        self.__optimizer = torch.optim.AdamW(self.parameters(),
                                             lr=learningRate,
                                             betas=betas,
                                             eps=eps,
                                             weight_decay=weightDecay)

        device = next(self.parameters()).device
        self.__dropout = nn.Dropout(self.__config.getDropoutRate()).to(device)

        Logging.GetInstance().Debug(f"MyModel initialized with config - {self.__config.getConfigType()}")

    def getOptimizer(self) -> torch.optim.Optimizer:
        return self.__optimizer

    def __getOptimizerParamsFromConfig(self):
        learningConfig = self.__config.getLearningConfig()
        learningRate = learningConfig.getLearningRate()
        if learningConfig.getBeta1() is not None and learningConfig.getBeta2() is not None:
            betas = (learningConfig.getBeta1(), learningConfig.getBeta2())
        else:
            betas = (0.9, 0.999)

        if learningConfig.getEpsilon() is not None:
            eps = learningConfig.getEpsilon()
        else:
            eps = 1e-8
        weightDecay = learningConfig.getWeightDecay()

        return learningRate, betas, eps, weightDecay

    def setOptimizerForFineTuning(self):
        Logging.GetInstance().Info("Setting optimizer for fine-tuning")
        learningRate, betas, eps, weightDecay = self.__getOptimizerParamsFromConfig()
        self.__optimizer = torch.optim.AdamW(
                                            filter(lambda p: p.requires_grad, self.parameters()),
                                            lr=learningRate,
                                            betas=betas,
                                            eps=eps,
                                            weight_decay=weightDecay)

    def setConfig(self, configFileName: str):
        Logging.GetInstance().Debug(f"Setting config from file: {configFileName}")
        self.__config = Config(configFileName)
        self.__config.loadConfig()

    def getConfig(self) -> Config:
        Logging.GetInstance().Debug("Getting config")
        return self.__config

    def setTokenizer(self, tokenizer: AutoTokenizer):
        Logging.GetInstance().Debug(f"Setting tokenizer: {tokenizer}")
        self.__tokenizer = tokenizer

    def setNumEpochs(self, numEpochs: int):
        Logging.GetInstance().Debug(f"Setting number of epochs: {numEpochs}")
        self.__numEpochs = numEpochs

    def setDataLoader(self, dataLoader: EncoderDecoderDataLoader):
        Logging.GetInstance().Debug(f"Setting data loader: {dataLoader}")
        self.__dataLoader = dataLoader

    def getDecoderStack(self) -> DecoderStack:
        return self.__decoderStack

    def getEncoderStack(self) -> EncoderStack:
        return self.__encoderStack

    def freezeOutputLayer(self):
        for param in self.__outputLayer.parameters():
            param.requires_grad = False
        Logging.GetInstance().Info("Output layer frozen for training")

    def unfreezeOutputLayer(self):
        for param in self.__outputLayer.parameters():
            param.requires_grad = True
        Logging.GetInstance().Info("Output layer unfrozen for training")

    def setValidationDataLoader(self, validationDataLoader: EncoderDecoderDataLoader):
        Logging.GetInstance().Debug(f"Setting validation data loader: {validationDataLoader}")
        self.__validationDataLoader = validationDataLoader

    def forward(self, inputIds: torch.Tensor, decoderInputIds: torch.Tensor) -> torch.Tensor:
        Logging.GetInstance().Debug(f"InputIds Shape : {inputIds.shape}")

        encoderOutput = self.__encoderStack(inputIds)
        Logging.GetInstance().Debug(f"Encoder output shape: {encoderOutput.shape}")

        decoderOutput = self.__decoderStack(decoderInputIds, encoderOutput)
        Logging.GetInstance().Debug(f"Decoder output shape: {decoderOutput.shape}")

        logits = self.__outputLayer(decoderOutput)
        Logging.GetInstance().Debug(f"Logits shape: {logits.shape}")

        logits = self.__dropout(logits)
        Logging.GetInstance().Debug(f"Logits after dropout shape: {logits.shape}")
        return logits

    def __applyRepetitionPenalty(self, logits: torch.Tensor) -> torch.Tensor:
        for batchIdx, prevTokens in enumerate(self.__generatedTokens):
            for tokenId in set(prevTokens):  # set() avoids penalizing multiple times
                if tokenId >= logits.size(-1):
                    continue

                # Get the original logit
                originalLogit = logits[batchIdx, tokenId].item()

                # Apply penalty according to the sign
                if originalLogit > 0:
                    logits[batchIdx, tokenId] /= self.__repetitionPenalty
                else:
                    logits[batchIdx, tokenId] *= self.__repetitionPenalty

        return logits

    def resetGeneratedTokens(self):
        Logging.GetInstance().Debug("Resetting generated tokens")
        self.__generatedTokens = [[]]

    def __softmaxWithTemperature(self, logits: torch.Tensor, temperature: float = 1.5, dim=-1, topK=3) -> torch.Tensor:

        logits = self.__applyRepetitionPenalty(logits)

        if topK > 0:
            # If topK is specified, we only keep the topK logits and set the rest to a very low value
            topKValues, _ = torch.topk(logits, topK, dim=-1)
            minTopKValue = torch.min(topKValues, dim=-1, keepdim=True).values
            logits[logits < minTopKValue] = float('-inf')

        if temperature > 0.0:
            # This function applies softmax with temperature scaling to the logits.
            # Temperature scaling is a technique used to control the randomness of predictions.
            # A higher temperature (>1) makes the distribution more uniform, while a lower temperature (<1) makes it sharper
            scaledLogits = logits / temperature
        else:
            scaledLogits = logits

        probabilityVector = torch.softmax(scaledLogits, dim=-1)
        Logging.GetInstance().Debug(f"Probability vector shape: {probabilityVector.shape}")

        return probabilityVector

    def getNextToken(self, logits: torch.Tensor) -> int:
        Logging.GetInstance().Debug(f"Input IDs shape: {logits.shape}")

        probabilityVector = logits[:, - 1, :]

        probabilityVector = self.__softmaxWithTemperature(
            probabilityVector,
            temperature=self.__config.getGeneratorConfig().getTemperature(),
            dim=-1,
            topK=self.__config.getGeneratorConfig().getTopK())
        if torch.any(torch.isnan(probabilityVector)) or probabilityVector.sum(dim=-1).eq(0).any():
            Logging.GetInstance().Debug("Invalid or empty probability vector, using uniform fallback.")
            probabilityVector.fill_(1.0 / probabilityVector.size(-1))

        #predictedToken = torch.argmax(probabilityVector, dim=-1, keepdim=True)
        predictedToken = torch.multinomial(probabilityVector, num_samples=1)

        self.__generatedTokens[0].append(predictedToken.item())

        Logging.GetInstance().Debug(f"Predicted token: {predictedToken}")

        return predictedToken

    def getLoss(self, logits: torch.Tensor, targetIds: torch.Tensor) -> torch.Tensor:
        Logging.GetInstance().Debug(f"Logit shape: {logits.shape}")

        lossFunction = torch.nn.CrossEntropyLoss(
            ignore_index=0,
            label_smoothing=self.__config.getLearningConfig().getLabelSmoothing())

        logitsFlat = logits.flatten(start_dim=0, end_dim=1)
        targetIdsFlat = targetIds.flatten(start_dim=0, end_dim=1)

        loss = (lossFunction(logitsFlat, targetIdsFlat))

        Logging.GetInstance().Debug(f"Loss: {loss.item()}")

        return loss

    def getPerplexity(self, loss) -> float:
        Logging.GetInstance().Debug(f"Calculating perplexity from loss: {loss.item()}")
        perplexity = torch.exp(loss)
        Logging.GetInstance().Debug(f"Perplexity: {perplexity}")
        return perplexity

    def loadParameters(self, loadFileName: str):
        loadFileHandler = open(loadFileName, "rb")
        missing_keys, unexpected_keys = self.load_state_dict(torch.load(loadFileHandler), strict=False)
        Logging.GetInstance().Info(f"Missing keys in loaded model: {missing_keys}")
        Logging.GetInstance().Info(f"Unexpected keys in loaded model: {unexpected_keys}")
        Logging.GetInstance().Info(f"Model parameters loaded from {loadFileName}")
        loadFileHandler.close()

    def validate(self):
        total_loss = 0
        total_tokens = 0
        vocabSize = 32128

        for inputIds, targetIds in self.__validationDataLoader.getDataLoader():
            device = next(self.parameters()).device
            inputIds = inputIds.to(device)
            targetIds = targetIds.to(device)

            # Shift targetIds to get decoder_input_ids
            decoderInputIds = torch.full_like(targetIds, 0)
            decoderInputIds[:, 1:] = targetIds[:, :-1]
            decoderInputIds[:, 0] = vocabSize + 2

            logits = self.__model(inputIds, decoderInputIds)

            # Flatten for loss (batch * seq_len)
            loss = self.getLoss(logits, targetIds)
            total_loss += loss.item() * targetIds.numel()
            total_tokens += targetIds.numel()
            Logging.GetInstance().Info(f"Validation Loss: {avg_loss:.4f}, total tokens: {total_tokens}")

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        Logging.GetInstance().Info(f"Validation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        return avg_loss, perplexity

    def getDataLoader(self):
        return self.__dataLoader.getDataLoader()

    def checkpoint(self,
                   saveFileName: str,
                   currentEpoch: int,
                   isLearningRateEnhancement: bool,
                   isCosineDecay: bool,
                   scheduler=None):
        if saveFileName:
            checkpointContent = {
                'modelStateDict': self.state_dict(),
                'optimizerStateDict': self.__optimizer.state_dict(),
                'schedulerStateDict': scheduler.state_dict() if scheduler else None,
                'config': self.__config,
                'isLearningRateEnhancement': isLearningRateEnhancement,
                'isCosineDecay': isCosineDecay,
                'currentEpoch': currentEpoch
            }
            torch.save(checkpointContent, saveFileName)

    def loadCheckpoint(self, loadFileName: str,
                       dataLoaderLength: int = 0,
                       isFineTuning: bool = False):
        from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
        if loadFileName:
            device = next(self.parameters()).device
            checkpoint = torch.load(loadFileName, map_location=device, weights_only=False)
            self.load_state_dict(checkpoint['modelStateDict'])
            if self.__config.getCheckpointConfig().useOptimiserStateFromCheckpoint():
                if isFineTuning:
                    self.__optimizer.load_state_dict(checkpoint['optimizerStateDict'])
                if self.__config.getCheckpointConfig().overRideOptimiserLrFromConfig():
                    for param_group in self.__optimizer.param_groups:
                        param_group['lr'] = self.__config.getLearningConfig().getLearningRate()
                    Logging.GetInstance().Info(f"Optimizer LR overridden to {self.__config.getLearningConfig().getLearningRate()}")

            scheduler: SequentialLR = None

            if self.__config.getCheckpointConfig().useLrWarmupStateFromCheckpoint() \
                and checkpoint['schedulerStateDict'] is not None \
                and dataLoaderLength > 0:
                total_steps = dataLoaderLength * self.__numEpochs
                warmup_steps = dataLoaderLength * self.__config.getLearningConfig().getWarmupStepPercent()
                warmup_steps = int(warmup_steps)

                Logging.GetInstance().Debug(f"Warmup steps: {warmup_steps}")
                cosine_steps = total_steps - warmup_steps

                # Warmup: linearly increase LR from 0 to 1x
                #warmup_scheduler = LambdaLR(
                #    self.__optimizer,
                #    lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0)
                #)
                #scheduler = warmup_scheduler
                #scheduler.load_state_dict(checkpoint['schedulerStateDict'])
            #self.__config = checkpoint['config']
            startEpoch = checkpoint['currentEpoch']
            if (isFineTuning == False) and self.__config.getCheckpointConfig().useLrWarmupStateFromCheckpoint():
                isLearningRateEnhancement = checkpoint.get('isLearningRateEnhancement')
            elif self.__config.getLearningConfig().isLrWarmupEnabled():
                isLearningRateEnhancement = True
            else:
                isLearningRateEnhancement = False

            if (isFineTuning == False) and self.__config.getCheckpointConfig().useCossineDecayStateFromCheckpoint():
                isCosineDecay = checkpoint.get('isCosineDecay')
            elif self.__config.getLearningConfig().isCosineDecayEnabled():
                isCosineDecay = True
            else:
                isCosineDecay = False

            Logging.GetInstance().Info(f"useLrWarmupStateFromCheckpoint: {self.__config.getCheckpointConfig().useLrWarmupStateFromCheckpoint()}, "
                                      f"isLrWarmupEnabled: {self.__config.getLearningConfig().isLrWarmupEnabled()}")

            Logging.GetInstance().Info(f"isLearningRateEnhancement: {isLearningRateEnhancement}, isCosineDecay: {isCosineDecay}")

            return startEpoch, scheduler, isLearningRateEnhancement, isCosineDecay

    def getCrossAttentionEntropy(self, epoch: int, count: int) -> torch.Tensor:
        return self.__decoderStack.getEntropy(epoch, count)

    def runTraining(self, saveFileName: str = "",
                    loadFileName: str = "",
                    isCheckpoint: bool = False,
                    isFineTuning: bool = False):
        from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
        vocabSize = 32128
        dataLoadLength = len(self.__dataLoader.getDataLoader())
        loadFileHandler = None
        isLearningRateEnhancement = self.__config.getLearningConfig().isLrWarmupEnabled()
        isCosineDecay = self.__config.getLearningConfig().isCosineDecayEnabled()
        scheduler = None

        if loadFileName:
            #loadFileHandler = open(loadFileName, "rb")
            #self.load_state_dict(torch.load(loadFileHandler))
            epoch, scheduler, isLearningRateEnhancement, isCosineDecay = self.loadCheckpoint(loadFileName,
                                                                                             dataLoadLength,
                                                                                             isFineTuning=isFineTuning)
            Logging.GetInstance().Info(f"Model loaded from {loadFileName}")

        if isLearningRateEnhancement:
            # Total training steps
            total_steps = dataLoadLength * self.__numEpochs
            warmup_steps = dataLoadLength * self.__config.getLearningConfig().getWarmupStepPercent()
            warmup_steps = int(warmup_steps)

            Logging.GetInstance().Debug(f"Warmup steps: {warmup_steps}")
            cosine_steps = total_steps - warmup_steps

            # Warmup: linearly increase LR from 0 to 1x
            warmup_scheduler = LambdaLR(
                self.__optimizer,
                lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0)
            )

            if isCosineDecay:
                # Cosine decay: decay LR from 1x to near-zero
                cosine_scheduler = CosineAnnealingLR(
                    self.__optimizer,
                    T_max=cosine_steps,
                    eta_min=self.__config.getLearningConfig().getLearningRate() * 0.1
                )

                # Combine both
                scheduler = SequentialLR(
                    self.__optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps]
                )
            else:
                scheduler = warmup_scheduler

        for epoch in range(self.__numEpochs):
            dataLoadLength = len(self.__dataLoader.getDataLoader())
            count = 0
            for inputIds, targetIds in self.__dataLoader.getDataLoader():
                self.__optimizer.zero_grad()
                device = next(self.parameters()).device
                inputIds = inputIds.to(device)
                targetIds = targetIds.to(device)
                # Shift targetIds to get decoder_input_ids
                decoderInputIds = torch.cat(
                    [torch.full((targetIds.size(0), 1), self.__tokenizer.pad_token_id, dtype=torch.long, device=device),
                    targetIds[:, :-1]],
                    dim=1
                )

                expectedOutputIds = targetIds

                logits = self(inputIds, decoderInputIds)

                # This piece of code is added to make the decoder more reliant on 
                # the encoder output, it is a hack to make the model more robust
                # and to avoid overfitting, it is not a standard practice
                #decoderDropoutScaling = 5
                #if count >= 1000:
                #    if (count // 500) % 2 == 0:
                #        decoderDropoutScaling = 4.5

                #for transportDecoder in self.__transformerBlocks:
                #    transportDecoder.setDropoutScaling(decoderDropoutScaling)
                self.__decoderStack.setDropoutScaling(1.0)  # Reset dropout scaling to 1.0

                layerFreezeEnabled = False
                if layerFreezeEnabled:
                    if epoch == 0 and count < 1000 and not self.__encoderStack.isFrozen():
                        # For the first 10% of the data, we use a higher dropout scaling
                        Logging.GetInstance().Info("Freezing decoder stack for first 1000 steps")
                        self.__encoderStack.freeze()
                    #    self.__decoderStack.freeze()
                    #elif epoch == 0 and count < 2000 and self.__encoderStack.isFrozen():
                    #    Logging.GetInstance().Info("Unfreezing decoder stack after first 1000 steps till 2000 steps")
                    #    self.__decoderStack.unfreeze()
                    elif not (epoch == 0 and count < 1000) and self.__encoderStack.isFrozen():
                        self.__encoderStack.unfreeze()

                # Calculating and printing token confidence -
                probs = torch.nn.functional.softmax(logits, dim=-1)
                tokenConfidence = probs.gather(dim=2, index=expectedOutputIds.unsqueeze(-1)).squeeze(-1)
                avgConf = tokenConfidence.mean().item()
 
                Logging.GetInstance().Debug(f"Logits shape: {logits.shape}")
                loss = self.getLoss(logits, expectedOutputIds)

                avgEntropy = 0.0
                if epoch > 0 or count > 0:
                    avgEntropy = self.__decoderStack.getEntropy(epoch=epoch, count=count)

                # Below section of code is to add entropy regularization to the
                # loss function. This is done to encourage the model to have
                # lower entropy in its cross attention weights
                if self.__config.getLearningConfig().getCrossAttRegRate() is not None:
                    isEntropyRegularizationEnabled = True
                else:
                    isEntropyRegularizationEnabled = False

                if isEntropyRegularizationEnabled:
                    loss = loss + self.__config.getLearningConfig().getCrossAttRegRate() * avgEntropy  # Adding entropy as a regularization term
                Logging.GetInstance().Debug(f"Loss: {loss.item()}")
                perplexity = self.getPerplexity(loss)
                loss.backward()
                # clipping the gradients to avoid exploding gradients
                # exploding gradients means the gradients become too large
                # and cause the model to diverge, to start with we are using
                # max_norm of 1.0, which is conservative and should work
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    max_norm=self.__config.getLearningConfig().getGradientClipping())

                Logging.GetInstance().Debug(f"Gradient Clipping - "
                                           f"{self.__config.getLearningConfig().getGradientClipping()}")

                self.__optimizer.step()

                if isLearningRateEnhancement:
                    # Step the scheduler if using learning rate enhancement
                    scheduler.step()
                    learningRateUsed = scheduler.get_last_lr()[0]
                else:
                    learningRateUsed = self.__optimizer.param_groups[0]['lr']

                if count % 100 == 0 and isCheckpoint and saveFileName:
                    #torch.save(self.state_dict(), saveFileName)
                    self.checkpoint(saveFileName, epoch, isLearningRateEnhancement, isCosineDecay, scheduler)
                    Logging.GetInstance().Info(f"Model checkpoint saved to {saveFileName}")

                Logging.GetInstance().Info(f"{count}/{dataLoadLength} [{epoch+1}/{self.__numEpochs}] "
                                           f"Loss: {loss:.2f} Perplexity: {perplexity:.2f}, "
                                           f"LR: {learningRateUsed:.6f}, "
                                           f"tokenConfidence: {avgConf:.4f}, "
                                           f"Cross-attn Entropy: {avgEntropy:.4f}")
                count += 1

            if isCheckpoint and saveFileName:
                self.checkpoint(saveFileName, epoch, isLearningRateEnhancement, isCosineDecay, scheduler)
                Logging.GetInstance().Info(f"Model checkpoint saved to {saveFileName}")
            Logging.GetInstance().Info(f"Epoch {epoch + 1}/{self.__numEpochs} completed")

        if not isCheckpoint and saveFileName:
            self.checkpoint(saveFileName, epoch, isLearningRateEnhancement, isCosineDecay, scheduler)

    def generateText(self, startingInput: torch.Tensor, maxNewToken: int) -> torch.Tensor:
        Logging.GetInstance().Debug(f"Starting inputs shape: {startingInput.shape}")
        inputs = startingInput.clone()

        batchSize, contextLength = startingInput.shape
        Logging.GetInstance().Debug(f"InputIds Shape : {startingInput.shape}")

        # Execute transformer encoder blocks

        device = next(self.parameters()).device
        inputs = inputs.to(device)

        encoderOutput = self.__encoderStack(inputs)
        Logging.GetInstance().Debug(f"Encoder output shape: {encoderOutput.shape}")
        vocabSize = 32128
        decoderInputIds = torch.tensor([[self.__tokenizer.pad_token_id]], device=startingInput.device)
        decoderInputIds = decoderInputIds.to(device)
        self.__generatedTokens = [[] for _ in range(batchSize)]  # Reset for repetition penalty

        for _ in range(maxNewToken):

            decoderOutput = self.__decoderStack(decoderInputIds, encoderOutput)
            Logging.GetInstance().Debug(f"Decoder inputs IDs shape: {decoderInputIds.shape}")

            logits = self.__outputLayer(decoderOutput)
            Logging.GetInstance().Debug(f"Logits shape: {logits.shape}")

            nextToken = self.getNextToken(logits)

            for i in range(batchSize):
                self.__generatedTokens[i].append(nextToken[i].item())

            # Update decoder inputs for next step
            decoderInputIds = torch.cat((decoderInputIds, nextToken), dim=1)
            inputs = torch.cat((inputs, nextToken), dim=1)

            # Break if any sentence reaches an end token
            if torch.any(nextToken == vocabSize + 1) or torch.any(nextToken == vocabSize + 3):
                Logging.GetInstance().Debug("Special token encountered {newToken}")
                break
            elif torch.any(nextToken == 0) or torch.any(nextToken == vocabSize + 2):
                Logging.GetInstance().Debug("Zero or start token encountered {newToken}")
                continue

        Logging.GetInstance().Debug(f"Final generated inputs shape: {inputs.shape}")
        self.resetGeneratedTokens()
        return inputs
