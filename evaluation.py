'''
Evaluation function for the classifier. Taken from the CS224N multitask classifier assignment, designed by Chris Manning.
'''

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np


TQDM_DISABLE = False

def model_eval(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels = batch['input_ids'],batch['attention_mask'], batch['label']
        b_labels_unbound = torch.unbind(b_labels, dim=1)

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)


        logits = model(b_ids, b_mask) # size [batch_size, ncodes]
        preds = torch.flatten((torch.sigmoid(logits) > 0.5).long()) # apply softmax, flatten, size [batch_size * ncodes]

        labels = torch.flatten(b_labels) # flatten

        y_true.extend(preds.numpy())
        y_pred.extend(labels.numpy())

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true