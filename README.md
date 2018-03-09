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

#### Data fields
id - a unique identifier for each sentence

text - some text written by one of the authors

author - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

A reminder about playground competitions: On Kaggle, the spirit of playground competitions is to have fun and learn together. Your score on the leaderboard doesn't earn you points, but you can still make it a rewarding competition for everyone by sharing your code in Kernels and contributing to Discussions (there are prizes for both!). In short, please don't look up the answers.

### Methods and Evaluation
#### 1 Traditional feature engineering and machine learning algorithms
In the first iteration of this project, traditional NLP techniques are used for the text classification task. Here are the steps:

(1) Tokenize the sentences using the WordNetLemmatizer. It is one of the NLTK stemming method. Lemmatize is better than pure stemming because we can have complete words in the tokenization result. However, in some cases, lemmatization has the drawback of failing to detect new words that are outside of the NLTK dictionary/lexicons. To simplify this problem, lemmatization is worth trying.

(2) Fit the tokenized words into Bag of Words model. This is actually done by the CountVectorizer using LemmaTokenizer in the first step.

(3) Feature selection. Not all the words are common enough to be used as a feature. If all the words are counted as a feature, then the dimension of the feature space will be too large, which can greatly slow down the next classification training steps. A Variance Threshold is set to 0.98 in this case. This leaves 161 features from the corpus.

(4) Classification and evaluation. Just simply use Sklearn library to do the classification. Random Forest is used in this case. After training, use cross validation to generate result reports.

##### Evaliation:

Random Forest:

Training Score: 0.985954338832

Confusion Matrix:
[[5568 1093 1239]
 [1784 2981  870]
 [2106 1084 2854]]

 10-fold Cross Validation Report:
             precision    recall  f1-score   support

          a       0.59      0.70      0.64      7900
          b       0.58      0.53      0.55      5635
          c       0.58      0.47      0.52      6044

avg / total       0.58      0.58      0.58     19579

ANN:
```
0.987026916594
[[4982 1368 1550]
 [1371 3222 1042]
 [1544  989 3511]]
             precision    recall  f1-score   support

          a       0.63      0.63      0.63      7900
          b       0.58      0.57      0.57      5635
          c       0.58      0.58      0.58      6044

avg / total       0.60      0.60      0.60     19579
```

The baseline of this project is about 40%(which is predict every label as the first author). The result gets about 58%/60%, which is better than random guess but still performs badly.


The traditional methods have the advantage of intuitive, fast and simple. It basically classify by 2 main points: 1. The appearance of certain words. 2. The appearance of certain combination of words.

Based on this characteristics, the sequence of words are lost. Using the CNN method in the second iteration is worth trying.



#### 2 Deep Learning: CNN in sentence classification
CNN is mostly used for computer vision tasks such as image recognition, but how this method can be used for NLP and why it works?

To be continued...






















"""End of file"""
