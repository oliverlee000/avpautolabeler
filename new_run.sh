#!/bin/bash
/usr/bin/python3 -m pip install --user pandas tokenizers==0.11.1 scikit-learn torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html transformers==4.29.2 safetensors==0.3.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
/usr/bin/python3 main.py --train data/train.csv --dev data/dev.csv --use_gpu