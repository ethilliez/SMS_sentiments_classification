import nltk
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

DATA_PATH = 'Data/'
ENCODING = 'TFIDF_Bag_of_words' # 'Bag_of_words' 

# Read data
SMS = pd.read_csv(DATA_PATH+'SMS_sentiment.csv')
SMS.pop('tweet_id')
SMS.pop('author')

# Clean data
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df
SMS = standardize_text(SMS, "content")

# Select two sentiments
SMS = SMS[(SMS.sentiment == 'happiness') | (SMS.sentiment == 'sadness')]
SMS = SMS.sample(frac=1).reset_index(drop=True)
print(SMS.head(4))

list_sms = SMS["content"].tolist()
list_sentiment = SMS["sentiment"].tolist()
# Encode ylabel as integer
le = preprocessing.LabelEncoder()
le.fit(list(set(list_sentiment)))
list_sentiment = le.transform(list_sentiment) 
# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(list_sms, list_sentiment, test_size=0.1, random_state=0)

if(ENCODING == 'Bag_of_words'):
# Bag of words encoding
    count_vectorizer = CountVectorizer()
    X_train_counts = count_vectorizer.fit_transform(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
elif(ENCODING == 'TFIDF_Bag_of_words'):
# Term Frequency, Inverse Document Frequency Bag of words
    tfidf_vectorizer = TfidfVectorizer()
    X_train_counts= tfidf_vectorizer.fit_transform(X_train)
    X_test_counts = tfidf_vectorizer.transform(X_test)

# Train classifier
clf = LogisticRegression(random_state=0)
clf.fit(X_train_counts, y_train)

# Test classifier
y_predicted = clf.predict(X_test_counts)
recall = recall_score(y_test, y_predicted)
accuracy = accuracy_score(y_test, y_predicted)
print('Accuracy: ', accuracy, " Recall: ", recall)
