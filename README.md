# SMS_sentiments_classification
Extract the sentiment from a set of text messages

## Description
Algorithm to predict the corresponding sentiment attached a SMS using supervised learning. The raw data can be found here: [https://www.crowdflower.com/data-for-everyone/](https://www.crowdflower.com/data-for-everyone/). The overall accuracy and recall obtained is 80%.

## Personal development goals:
- Practising basic NLP techniques (various bag-of-words and word2vec encodings).
- Practising supervised classification technique with a simple Logistic Regression using [sklearn](https://github.com/scikit-learn/scikit-learn).

## Status of development:
- Data pre-processing implemented
- Bag-of-words encoding implemented
- Term Frequency Inverse Document Frequency Bag-of-words encoding implemented
- Logistic Regression training implemented
- Logistic Regression testing implemented
- Word2vec to COME

## Requirements:
The main librairies required are: `numpy` and `nltk`,`sklearn`. They can be installed using `pip install` or `conda install`.

## Execution:
`python3 main.py`
