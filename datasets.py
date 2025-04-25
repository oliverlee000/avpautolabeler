from transformers import BertTokenizer
import torch, re
import csv
from torch.utils.data import Dataset

'''
This just contains the TextDataset class, the only relevant data type for this project.
'''

class TextDataset(Dataset):
    def __init__(self, texts, labels, ncodes):
        self.texts = texts
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.ncodes = ncodes

        labels_as_tuple = [list(map(int, re.findall(r'[0-9]+', l))) for l in labels]

        # Initialize tensor of shape (len(labels), ncodes) with zeros
        self.labels = torch.zeros((len(labels_as_tuple), self.ncodes), dtype=torch.int32)

        # Fill in the presence indicators
        for i, label_list in enumerate(labels_as_tuple):
            for label in label_list:
                if label >= self.ncodes or label < 0:
                    raise ValueError(f"Invalid label value:\n{label} when number of codes is {self.ncodes}")
                self.labels[i, label] = 1
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return {"input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label": label}

