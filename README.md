Trains and runs an autolabeling model based on a dataset, split into train, dev, and test sets.

To test on actual dataset, run command
```
python main.py --ncodes 5 --train example_data/train.csv --dev data/dev.csv
```

To test on example dataset, run command
```
python main.py --ncodes 5 --train example_data/train.csv --dev example_data/dev.csv --fine-tune-mode last-linear-layer
```

Call command ```python main.py --help``` to see all possible parameters.

**(Below is more for me than anyone else to preprocess the data so that it can be fed into the model)**

Since I'm just looking to improve the model now, the code does not run the model on the test set, just the train and dev sets.

Dataset should be labeled and stored as CSV files in the folder "data":
- data/train.csv: training set
- data/dev.csv: dev set
- data/test.csv: test set

Each of these csvs should contain a column called "text" (a string representing the text of a given code) and "label" (a string containing a list of numbers from [0, K), based on the K codes in the dataset, deliminated by commas).

(See example_data folder for example format)
- For example, if the sentence "I went to Stanford University" had the codes 1, 3, 4 (representing three codes enumerated 1, 3, and 4), then this row would have the following values:
  - "text": "I went to Stanford University"
  - "label": "1, 3, 4"
