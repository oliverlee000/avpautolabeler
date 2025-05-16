#!/bin/bash
env -i PATH=/usr/bin HOME=$HOME \
    /usr/bin/python3 -m pip install --user pandas tokenizers==0.20.0 scikit-learn torch==1.10.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    /usr/bin/python3 main.py --train data/train.csv --dev data/dev.csv --use_gpu