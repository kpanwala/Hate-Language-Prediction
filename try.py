# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train_tweets.csv')

y=dataset['label']
x=dataset['tweet']

y=pd.get_dummies(y[:])

# nltk.download('stopwords')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ',x.iloc[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
cp=corpus

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer1.pickle", "wb"))

cvv = pickle.load(open("tfidf_vectorizer1.pickle", "rb"))

X=cvv.transform(cp)

# del(classifier)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

classifier = load_model('cnn_from_hate1.model')
classifier.load_weights('cnn_from_hate1.hdf5')

import pickle

pickle_in = open("y_test_hate1.pickle","rb")
y_test = pickle.load(pickle_in)


# Predicting the Test set results
y_pred = classifier.predict(X)

y_pred1=[]
# y_pred1 = y_pred.apply(lambda x: myfunc(y_pred[:,0:1], y_pred[:,1:2],y_pred[:,2:3]), axis=1)
for i in range(len(y_pred)):
    if y_pred[i,0] > y_pred[i,1]:
        y_pred1.append({0:1,1:0})
    else:
        y_pred1.append({0:0,1:1})
    
y_pred1=pd.DataFrame(y_pred1)

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

psa=precision_score(y, y_pred1, average='macro')    # 70.13%
rsa=recall_score(y, y_pred1, average='macro')   #  92.03%



# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
classifier_random1 = RandomForestRegressor(n_estimators = 5, random_state = 0)
classifier_random1.fit(X_train, y_train)


# Predicting a new result
y_pred_rand1 = classifier_random1.predict(X_test)
y_pred2=y_pred_rand1
y_pred2=y_pred2.round()
y_pred_rand1=y_pred2

from sklearn.metrics import precision_score
psr=precision_score(y_test, y_pred_rand1, average='macro')  #80.69S%
rsr=recall_score(y_test, y_pred_rand1, average='macro')    #80.6%


import pickle


with open('hate_model_rf.pickle', 'wb') as f:
    pickle.dump(classifier_random1, f)



with open('hate_model_rf1.pickle', 'rb') as f:
    rf = pickle.load(f)


preds = rf.predict(X_test)
preds=preds.round()


