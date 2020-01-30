import pickle

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


cvv = pickle.load(open("tfidf_vectorizer.pickle", "rb"))

cpp=[]
review = re.sub('[^a-zA-Z]', ' ', "Muslims should be brutally killed")
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
cpp.append(review)

ans=cvv.transform(cpp).toarray()


from keras.models import load_model
model1 = load_model('cnn_from_hate.model')

model1.load_weights('cnn_from_hate.hdf5')

y_pred_ans=[]
y_pred1=[]
y_pred_ans = model1.predict(ans)

if y_pred_ans[0,0] > y_pred_ans[0,1] and y_pred_ans[0,0] > y_pred_ans[0,2]:
    y_pred1.append(0)
elif y_pred_ans[0,2] > y_pred_ans[0,0] and y_pred_ans[0,2] > y_pred_ans[0,1]:
    y_pred1.append(2)
else:
    y_pred1.append(1)
    
if y_pred1==0:
    my_pred="hate speech"
elif y_pred1==[1]:
    my_pred="offensive"
else:
    my_pred="normal speech"