import argparse, np, random, tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel

'''
Here we define the classifier based on the BERT base model.
'''

BERT_HIDDEN_SIZE = 768
NUMBER_OF_CODES = 10

# Dropout probabilities
INPUT_DROP = 0.1
HIDDEN_DROP = 0.4
OUTPUT_DROP = 0.0


'''
Consists exclusively of a feed forward layer
'''
class FF(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_prob=HIDDEN_DROP, relu=False):
        super().__init__()
        # Feed forward.
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, output_size)
        self.af = F.gelu
        if relu:
            self.af = F.relu


    def forward(self, hidden_states, activation=True):
        """
        Put elements in feed forward.
        Feed forward consists of:
        1. a dropout layer,
        2. a linear layer, and
        3. an activation function.

        If activation = True, use activation
        """
        hidden_states = self.dropout(hidden_states)
        output = self.dense(hidden_states)
        if activation:
            output = self.af(output)
        return output

# Define classification model
class BertClassifier(nn.Module):
    def __init__(self, config):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # Allow the weights in the BERT model to be adjusted if fine_tune_mode == full-model, else fix them
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True

        # Create series of feed forward layers
        linear_layers = nn.ModuleList()
        if config.num_linear_layers > 1:
            linear_layers.append(FF(BERT_HIDDEN_SIZE, config.ll_hidden_size, INPUT_DROP))
            linear_layers.extend([FF(config.ll_hidden_size, config.ll_hidden_size, HIDDEN_DROP) for _ in range(config.num_linear_layers - 2)])
            linear_layers.append(FF(config.ll_hidden_size, NUMBER_OF_CODES, OUTPUT_DROP))
        else:
            linear_layers.append(FF(BERT_HIDDEN_SIZE, NUMBER_OF_CODES))
        self.linear_layers = linear_layers
    
    # Predicts code given an input
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeds = outputs.last_hidden_state[:, 0, :]  # Use CLS token representation
        for i, layer_module in enumerate(self.linear_layers[:-1]):
            embeds = layer_module(embeds, activation=True)
        output = self.linear_layers[-1](embeds, activation=False)
        return output