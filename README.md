Introduction
------------

This repository contains a from-scratch implementation of a Large Language Model (LLM) based on a sequence-to-sequence encoder–decoder Transformer architecture. The project is designed for deep learning research, architectural experimentation, pre-training, fine-tuning, and educational exploration of modern Transformer-based language models.

Unlike high-level wrapper implementations, this project focuses on low-level architectural clarity, allowing full control over:

Token embeddings
Positional embeddings
Multi-head attention
Feed-forward layers
Layer normalization
Training loops
Optimization and loss computation

Key Highlights
--------------

- From-scratch Encoder–Decoder Transformer implementation
- Custom Attention, FFN, and LayerNorm layers
- Config-driven architecture (fully modular design)
- Supports pre-training & fine-tuning
- Works with JSON-based input–target datasets
- Custom training loop and inference pipeline
- Research-friendly structure for experimentation
- CPU training supported (for demonstrations)

Model Architecture
-------------------

- Architecture Type: Sequence-to-Sequence Transformer
- Encoder: Multi-layer self-attention + feedforward blocks
- Decoder: Masked self-attention + cross-attention + feedforward blocks
- Normalization: Pre-Norm Layer Normalization
- Attention: Multi-Head Self & Cross Attention
- Training Objective: Supervised Seq2Seq Language Modeling (Using denoising method)

Project Structure
-----------------

├── Common/                 # Common functionalities like logging

├── LlmGenAi/               # LLM Gen AI Functionalities

    ├── Config/             # Model & training configuration
    
    ├── DataPreperation/    # DataSet and Dataloader
    
    ├── TextGeneration/     # Text Generation
    
        ├── LearningDataSet # Datasets for training
        
        ├── Parameters      # Pretrained Parameters
        
        ├── Src             # Model and Entry point for the model
        
            ├── StartLlm.py # Entry point to the model
            
    ├── Transformers/       # Transformer Encoder and Decoder stacks
└── README.md

Dataset Format
--------------

- This is the encoder-decoder based model, since encoder layer is involved, so denoising method is used for pretraining.
- Here 2 types of JSONs are supported
  1. Json with array of input target pairs like below -
     [
      {
      "Input": "How <mask> doing?",
      "Target": "How are you doing?"
     }
     ]
     Here the input is having span masking
  2. Json with array of text like below and the span masking is done by dataloader. The span masking is done at start of every epoch hence the mask positions keep on changing every epoch -
    [
      {
        "Text": "How are you doing"
      },
    ]

Training
--------

- To train on single GPU / CPU -

python3 StartLlm.py --device "cuda" --train <dataset> --saveparam <file name with path to save parameters> --epoch <number of epochs to train on the given dataset>

- To train on multiple GPUs -
  - In Config file Update "llmStartConfig" parameters. Please refer to LlmGenAi/Config/Test/TestConfig.json
  - Run below command from LlmGenAi/TextGeneration/Src

  torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 StartLlm.py

Inference
---------

- To run the LLM run below command at LlmGenAi/TextGeneration/Src

  python3 StartLlm.py --device "cpu" --runllm --loadparams <File name with path to the parameters file>

Disclaimer
----------

This is a research and educational implementation, not a production-grade deployment framework. Performance and memory optimizations are intentionally secondary to architectural clarity.


License
-------

MIT License — free to use for research and educational purposes.

Author
-----

*Mayank Arora*

Telecom Engineer and AI / ML Enthusiast
