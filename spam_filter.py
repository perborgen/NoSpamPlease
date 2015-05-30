# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
stemmer = SnowballStemmer('english')
stopWords = stopwords.words('english')
vectorizer = CountVectorizer(analyzer="word")
example = "Thanks for your subscription to Ringtone UK your mobile will be charged £5/month Please confirm by replying YES or NO. If you reply NO you will not be charged"

data = pd.read_csv('data/data.csv',header=0)

data_texts = data["text"]

data_labels = data["label"]
data_labels = np.array(data_labels)
data_labels = [0 if x == "spam" else 1 for x in data_labels]
data_labels = np.array(data_labels)

def cleanUp(texts):
	clean_texts = []
	for text in texts:
		text = re.sub("[^a-zA-Z]", " ", text)
		stemmed_text = []
		for word in text.split():
			stemmed_word = stemmer.stem(word)
			stemmed_text.append(stemmed_word)
		meaningful_text = [word for word in stemmed_text if not word in stopWords]
		clean_texts.append(" ".join(meaningful_text))
	return clean_texts


def vectorize(text):
	vectorzed_data = vectorizer.fit(text)
	vectorzed_data = vectorizer.transform(text)
	vectorzed_data = vectorzed_data.toarray()
	return vectorzed_data



def sliceData(data,labels,num_train):
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []

	length = len(data)
	for i in xrange(0,length):
		if i < num_train:
			train_data.append(data[i])
			train_labels.append(labels[i])
		elif i >= num_train:
			test_data.append(data[i])
			test_labels.append(labels[i])
	return {'train_data': train_data, 'train_labels': train_labels, 'test_data': test_data, 'test_labels': test_labels}



clean_data = cleanUp(data_texts)
vectorized_data = vectorize(clean_data)

print 'vectorized_data[0]: ', vectorized_data[0]
print len(vectorized_data[0])
all_data = sliceData(vectorized_data[0:500],data_labels,400)


train_data = np.array(all_data['train_data'])
train_labels = np.array(all_data['train_labels'])
test_data = np.array(all_data['test_data'])
test_labels = np.array(all_data['test_labels'])


# -------------> KNN with 3000 training set, no extra features - score:  0.912519440124
#knn = neighbors.KNeighborsClassifier()
#print 'about to fit'
#knn.fit(train_data, train_labels)
#score = knn.score(test_data,test_labels)
#print 'score: ', score


# --------------> RandomForest with 3000 training set, no extra features - score: 0.96967340591
#forest = RandomForestClassifier(n_estimators=100)
#print 'about to fit'
#forest = forest.fit(train_data,train_labels)

#length_of_test = len(test_labels)
#score = 0
#for n in xrange(length_of_test):
#	prediction = forest.predict(test_data[n])
#	prediction = prediction[0]
#	correct_answer = test_labels[n]
#	correct_answer = int(correct_answer)
#	if int(prediction) == int(correct_answer):
#		score += 1
#	score_normalized = float(score) / float(length_of_test)
#print 'score_normalized: ', score_normalized










## FEATURE ENGINEERING

#FEATURES = ["char_length", "upperCaseWords"]


#char_length = [len(x) for x in texts]
upperCaseWords = []
## find upper case words longer than 1 character
for text in data_texts[0:100]:
	count = 0
	words = text.split()
	for word in words:
		if word.isupper():
			if len(word) > 1:
				count +=1
				#print 'upperword: ', word
				#print 'text: ', text
	length_of_word = len(words)
	upperCaseRatio = float(count) / float(length_of_word)
	upperCaseWords.append(upperCaseRatio)

print upperCaseWords


