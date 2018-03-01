# Spooky_Author_Identification
Try some interesting text-classification algorithms.

### Desctiption
The development of this project follows the instructions of a Kaggle competition during the Halloween of 2017.

The link of the competition is https://www.kaggle.com/c/spooky-author-identification/ .

### Intro of Data from Kaggle

The competition dataset contains text from works of fiction written by spooky authors of the public domain: Edgar Allan Poe, HP Lovecraft and Mary Shelley. The data was prepared by chunking larger texts into sentences using CoreNLP's MaxEnt sentence tokenizer, so you may notice the odd non-sentence here and there. Your objective is to accurately identify the author of the sentences in the test set.

#### File descriptions
train.csv - the training set

test.csv - the test set

sample_submission.csv - a sample submission file in the correct format

####Data fields
id - a unique identifier for each sentence

text - some text written by one of the authors

author - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

A reminder about playground competitions: On Kaggle, the spirit of playground competitions is to have fun and learn together. Your score on the leaderboard doesn't earn you points, but you can still make it a rewarding competition for everyone by sharing your code in Kernels and contributing to Discussions (there are prizes for both!). In short, please don't look up the answers.
