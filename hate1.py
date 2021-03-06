# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train_tweets.csv')


dataset1=dataset.where(dataset['label']==1)
dataset2=dataset.where(dataset['label']==0)

dataset1=dataset1.dropna(how='all')
dataset2=dataset2.dropna(how='all')

dataset2=dataset2.sample(n = 2242)

main_data=dataset1.append(dataset2)

main_data=main_data.sample(n = 4484)


y=main_data['label']


Y=np.array(y)
Y=Y.reshape(-1,1)
X=main_data['tweet']

X=np.array(X)
X=X.reshape(-1,1)
X=X.astype(str)

y=pd.get_dummies(y[:])

# nltk.download('stopwords')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

# tp=X.iloc[0]

for i in range(0, len(main_data)):
    review = re.sub('[^a-zA-Z]', ' ',X.iloc[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
cp=corpus

from sklearn.feature_extraction.text import TfidfVectorizer

""" 
# settings that you use for count vectorizer will go here
tfidf_vectorizer=TfidfVectorizer(use_idf=True)
 
# just send in all your docs here
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(cp)

# get the first vector out (for the first document)
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]
 
# place tf-idf values in a pandas data frame
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
df=df.sort_values(by=["tfidf"],ascending=False)

"""

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2,stop_words='english')
# TF-IDF feature matrix
X1 = tfidf_vectorizer.fit_transform(corpus).toarray()

import pickle
pickle.dump(cv, open("count_vectorizer1.pickle", "wb"))


import pickle
pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer1.pickle", "wb"))

cvv = pickle.load(open("tfidf_vectorizer1.pickle", "rb"))
# X=cvv.fit_transform(cp)
"""
y=np.array(y)
y=y.reshape(-1,1)
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.2, random_state = 0)

# del(classifier)

import keras
from keras.models import Sequential
from keras.layers import Dense


input_dim = X_train.shape[1]

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim =2,init = 'uniform',  activation = 'relu', input_dim = input_dim))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 2, init = 'uniform',  activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 2, init = 'uniform',  activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath = 'cnn_from_hate1.hdf5', verbose = 1, save_best_only = True)


history = classifier.fit(X_train,y_train,
        batch_size = 10,
        epochs=15,
        validation_data=(X_test, y_test),
        callbacks = [checkpointer],
        verbose=2, shuffle=True)

from matplotlib import pyplot as plt

"""  
# AS we have metrics as loss so below portion may not work
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
"""

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# from keras.models import save_weights
# classifier.save_weights('cnn_from_hate_weights1.hdf5')
classifier.save('cnn_from_hate1.model')

import pickle

pickle_out = open("x_train_hate1.pickle","wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()


pickle_out = open("y_train_hate1.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()


pickle_out = open("x_test_hate1.pickle","wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test_hate1.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()


pickle_in = open("y_test_hate1.pickle","rb")
y_test = pickle.load(pickle_in)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred1=[]
# y_pred1 = y_pred.apply(lambda x: myfunc(y_pred[:,0:1], y_pred[:,1:2],y_pred[:,2:3]), axis=1)
for i in range(len(y_pred)):
    if y_pred[i,0] > y_pred[i,1]:
        y_pred1.append({0:1,1:0})
    else:
        y_pred1.append({0:0,1:1})
    
y_pred1=pd.DataFrame(y_pred1)


# y_pred2=pd.get_dummies(y_pred1[0])

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

psa=precision_score(y_test, y_pred1, average='macro')    # 87.14%
rsa=recall_score(y_test, y_pred1, average='macro')   #  86.89%




# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
gauss = GaussianNB()
gauss.fit(X_train.astype(int), y_train.astype(int))

# Predicting the Test set results
y_pred_gauss = gauss.predict(X_test.astype(int))

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

psg=precision_score(y_test, y_pred_gauss, average='macro')    # 51.89%
rsg=recall_score(y_test, y_pred_gauss, average='macro')   #  33.75%

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

"""
rf = RandomForestRegresor()
rf.fit(X, y)
"""

with open('hate_model_rf.pickle', 'wb') as f:
    pickle.dump(classifier_random1, f)



with open('hate_model_rf.pickle', 'rb') as f:
    rf = pickle.load(f)


preds = rf.predict(X_test)
preds=preds.round()


