from transformers import BertTokenizer
import torch, re
import csv
from torch.utils.data import Dataset

'''
This just contains the TextDataset class, the only relevant data type for this project.

- For labels, assumes that the value in 'labels' is a string with all labels delimited by commas.

For each value in labels, creates a vector (size K, where K is no. of codes) storing the codes present.
- Each dimension in the vector represents a code, with the Kth dimension representing null code (when a sentence
has no code assigned to it).
'''

class TextDataset(Dataset):
    def __init__(self, texts, labels, ncodes):
        self.texts = texts
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.ncodes = ncodes

        # Initialize tensor of shape (n, K) with zeros
        self.labels = torch.zeros((len(labels), self.ncodes), dtype=torch.int32)

        # Split each value in 'labels' column to list of all codes
        regex_pattern = r'[0-9]+'

        for i in range(len(labels)):
            # Find all codes, delimited by some non-numeric character
            # i.e. '1, 2, 3' -> [1, 2, 3]
            all_codes = list(map(int, re.findall(regex_pattern, labels[i])))

            # Iterate through codes, storing each code into corresponding dimension in labels vector

            # If a sentence has no codes, assign it the Kth value in labels vector
            for code in all_codes:
                if code >= self.ncodes or code < 0:
                    raise ValueError(f"Invalid label value:\n{code} when number of codes is {self.ncodes}")
                self.labels[i, code] = 1
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return {"input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label": label}

