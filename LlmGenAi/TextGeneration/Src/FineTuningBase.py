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
from abc import ABC, abstractmethod

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

class FineTuningBase(nn.Module):
    _config: Config
    _device: str
    _model: llm
    _isFrozen: bool = False  # Flag to check if the encoder stack is frozen

    def __init__(self, config: Config, model: llm, device: str):
        super(FineTuningBase, self).__init__()
        self._config = config
        self._device = device
        self._model = model

        Logging.GetInstance().Info("FineTuningBase model initialized successfully")

    @abstractmethod
    def freezeModel(self):
        raise NotImplementedError("Method to be implemented by derived class")