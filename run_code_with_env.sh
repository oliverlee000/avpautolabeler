#!/bin/bash
env -i PATH=/usr/bin HOME=$HOME \
    /usr/bin/python3 -m pip install --user pandas tokenizers scikit-learn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    /usr/bin/python3 main.py --train data/train.csv --dev data/dev.csv --use_gpu