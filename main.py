import argparse, random, torch, tqdm, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models import CodingClassifier, BertClassifier
from datasets import TextDataset
from evaluation import model_eval
from tqdm import tqdm

BERT_HIDDEN_SIZE = 768

# Load BERT tokenizer and model

'''
load_data_from_csv
Helper function, reads csv file, returns texts and labels
Use to prepare csv file to convert into TextDataset objects

ASSUMES:
Two columns in the csv file that are called 'text' and 'label'
'''
# Reads csv file, returns texts and labels
def load_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    return texts, labels

'''
batch_data
Helper function, return DataLoader given pathname of dataset.

ASSUMES:
Two columns in the csv file that are called 'text' and 'label'
'''
def batch_data(args, path):
    texts, labels = load_data_from_csv(path)
    data = TextDataset(texts, labels, args.ncodes)
    dataloader = DataLoader(data, batch_size= args.batch_size, shuffle=True)
    return dataloader

'''
save_model
Helper function, saves model into local computer
'''
def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

'''
seed_everything
Helper function, fixes seed
'''
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

'''
train_classifier
Trains classifier over several epochs (specified by args.epochs), then assesses accuracy based on devset
Saves the model that performs the best on the dev set

ASSUMES:
Data is split into a training and dev set, in two different files (args.train, args.dev)
Data is in csv format
'''
def train_classifier(args):
    # Prepare training and dev set dataloaders
    train_dataloader, dev_dataloader = batch_data(args, args.train), batch_data(args, args.dev)

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Prepare config for determining model settings
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'll_hidden_size': args.hidden_size,
              'num_linear_layers': args.num_linear_layers,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode,
              'ncodes': args.ncodes
    }
    config = SimpleNamespace(**config) # Necessary to allow you to access properties via .

    # Initialize model and optimizer
    classifier_model = CodingClassifier(config).to(device)
    loss_fn = F.binary_cross_entropy_with_logits
    optimizer = optim.Adam(classifier_model.parameters(), lr=args.lr)
    # Run the training loops
    classifier_model.train()
    num_batches = 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device) # size [batch_size, ncodes]
            
            optimizer.zero_grad()
            logits = classifier_model(input_ids, attention_mask)  # size [batch_size, ncodes]

            # Go through each code, get training loss for each code
            loss = 0
            for i in range(args.ncodes):
                loss += loss_fn(logits[:, i], label[:, i].float())
            loss.backward()
            optimizer.step()
            num_batches += 1
            epoch_loss += loss.item()
        train_loss = epoch_loss / num_batches
        best_dev_acc = 0
        current_dev_acc, _, _, _, _ = model_eval(dev_dataloader, classifier_model, device)
        # Save model if current dev accuracy is better than best dev accuracy
        if current_dev_acc > best_dev_acc:
            best_dev_acc = current_dev_acc
            save_model(classifier_model, optimizer, args, config, args.filepath)
        print(f"Epoch {epoch+1}: train loss :: {train_loss :.3f}, dev acc :: {current_dev_acc :.3f}")
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

    print("Training complete!")

''''
test_classifier
Opens the saved model,, then runs it on the dev set
Creates an output file that lists its accuracy on the dev set and predicted codes.
'''
def test_classifier(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = CodingClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        dev_dataloader = batch_data(args, args.dev)
        dev_acc, dev_f1, dev_y_pred, dev_y_true, dev_ids = model_eval(dev_dataloader, model, device)
        # CHANGE to be a CSV with sentence andmodel's prediction
        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            f.write(f"id \t Predicted_Codes \n")
            for p, s in zip(dev_ids, dev_y_pred):
                f.write(f"{p} , {s} \n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int,
                        default = 50)
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="full-model")
    parser.add_argument("--use_gpu", action='store_true')

    # Replace with pathnames of actual train, dev, test data
    parser.add_argument("--train", type=str, default="data/train.csv")
    parser.add_argument("--dev", type=str, default="data/dev.csv")
    parser.add_argument("--test", type=str, default="data/test.csv")
    parser.add_argument("--eval", type=str, help="\'y\' to run eval model, \'n\' else", default='n')

    parser.add_argument("--dev_out", type=str, default="predictions/dev-output.csv")
    parser.add_argument("--test_out", type=str, default="predictions/test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    parser.add_argument("--num_linear_layers", type=int, default=1)

    parser.add_argument("--ncodes", type=int, help="Number of codes in the dataset", default=20)

    args = parser.parse_args()
    return args

def print_args(args):
    print("Parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

if __name__ == "__main__":
    args = get_args()
    print_args(args)
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-classifier.pt' # Save path.
    if args.eval == 'y':
        test_classifier(args)
    else:
        seed_everything(args.seed)  # Fix the seed for reproducibility.
        train_classifier(args)
        test_classifier(args)