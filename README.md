Trains and runs an autolabeling model based on a dataset, split into train, dev, and test sets.

Since I'm just looking to improve the model now, the code does not run the model on the test set, just the train and dev sets.

Autocoding label is a classification model, taking a string of text as input, feeding it into an embedding layer (pretrained from Bert), then into a feed forward layer. 

Dataset should be labeled and stored as CSV files in the folder "data":
- data/train.csv: training set
- data/dev.csv: dev set
- data/test.csv: test set

Columns for each of the csv file should be "text" (containing the text of a given code) and "label" (containing the label of the code).
