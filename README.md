Trains and runs an autolabeling model based on a dataset, split into train, dev, and test sets.

Since I'm just looking to improve the model now, the code does not run the model on the test set, just the train and dev sets.

Autocoding label is a classification model, taking a string of text as input, feeding it into an embedding layer (pretrained from Bert), then into a feed forward layer. 

Dataset should be labeled and stored as CSV files in the folder "data":
- data/train.csv: training set
- data/dev.csv: dev set
- data/test.csv: test set

Each of these csvs should contain a column called "text" (a string representing the text of a given code) and "label" (a string containing a list of numbers from [0, K), based on the K codes in the dataset, deliminated by commas).
- For example, if the sentence "I went to Stanford University" had the codes 1, 3, 4 (rrepresenting three codes enumerated 1, 3, and 4), then the "text" column would contain "I went to Stanford Univeristy" and the label column would contain "1, 3, 4"
