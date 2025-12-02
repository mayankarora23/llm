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
from MyModelImplement import MyModel as llm
from FineTuningBase import FineTuningBase

class FineTuningLora(FineTuningBase):

    def __init__(self, config: Config, model: llm, device: str):
        super(FineTuningLora, self).__init__(config, model, device)
        model.setOptimizerForFineTuning()

        Logging.GetInstance().Info("FineTuningLora model initialized successfully")

    def freezeModel(self):
        if self._isFrozen:
            Logging.GetInstance().Warning("Model is already frozen")
            return

        # Freeze all parameters in the model
        for param in self._model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA layers
        for encoderBlock in self._model.getEncoderStack().getTransformerEncoderBlocks():
            encoderBlock.getMultiHeadAttention().unfreezeQuery()
            encoderBlock.getMultiHeadAttention().unfreezeValue()

        for decoderBlock in self._model.getDecoderStack().getTransformerDecoderBlocks():
            decoderBlock.getMaskedMultiHeadAttention().unfreezeQuery()
            decoderBlock.getMaskedMultiHeadAttention().unfreezeValue()
            decoderBlock.getCrossMultiHeadAttention().unfreezeQuery()
            decoderBlock.getCrossMultiHeadAttention().unfreezeValue()

        self._isFrozen = True

    def runTraining(self, saveFileName: str = "", loadFileName: str = "", isCheckpoint: bool = False):
        if not self._isFrozen:
            Logging.GetInstance().Warning("Model is not frozen. Freezing the model before training.")
            self.freezeModel()
            trainable_params = [n for n, p in self._model.named_parameters() if p.requires_grad]
            Logging.GetInstance().Info(f"Number of trainable parameters after freezing: {len(trainable_params)}")
        self._model.runTraining(saveFileName, loadFileName, isCheckpoint, isFineTuning=True)
