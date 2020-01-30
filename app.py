from flask import Flask,render_template,url_for,request
import tensorflow
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.models import load_model


app = Flask(__name__,template_folder='template')

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    cvv = pickle.load(open("tfidf_vectorizer.pickle", "rb"))
    model1 = load_model('cnn_from_hate.model')
    model1.load_weights('cnn_from_hate.hdf5')
    y_pred_ans=[]
    y_pred1=[]
    cpp=[]  #
    if request.method == 'POST':
        message = request.form['message']
        review = re.sub('[^a-zA-Z]', ' ',message)  
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        cpp.append(review)
        ans=cvv.transform(cpp).toarray()
        y_pred_ans = model1.predict(ans)

        if y_pred_ans[0,0] > y_pred_ans[0,1] and y_pred_ans[0,0] > y_pred_ans[0,2]:
            y_pred1.append(0)
        elif y_pred_ans[0,2] > y_pred_ans[0,0] and y_pred_ans[0,2] > y_pred_ans[0,1]:
            y_pred1.append(2)
        else:
            y_pred1.append(1)
                
        if y_pred1==[0]:
            my_pred=0
        elif y_pred1==[1]:
            my_pred=1
        else:
            my_pred=2
    return render_template('result.html',prediction = my_pred,l=y_pred_ans)



if __name__ == '__main__':
	app.run(debug=False)