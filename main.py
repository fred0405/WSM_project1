from __future__ import division, unicode_literals
import math
import nltk
from textblob import TextBlob as tb
from nltk.corpus import stopwords #remove stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer #stemming
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import sys

def tf(key, blob):
    sum = 0
    for i in blob: 
        sum = sum + blob[i]
    return blob[key] / sum

def tf_calculate(key, blob, sum):
	return blob[key] / sum

def n_containing(key, bloblist):
    return sum(1 for blob in bloblist if key in blob.keys())

def idf(key, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(key, bloblist)))

def tfidf(key, blob, bloblist):
    return tf(key, blob) * idf(key, bloblist)

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = data.replace(symbols[i], ' ')
        data = data.replace("  ", " ")
    data = data.replace(',', '')
    return data

def remove_apostrophe(data):
    return data.replace("'", "")

def Cosine_Similarity(Query,Document):
	return dot_product(Query,Document) / (distance(Query) * distance(Document))

def dot_product(Query,Document):
	sum = 0
	for i in range(len(Document)):
		sum = sum + (Query[i] * Document[i])
	return sum

def distance(Vec):
	sum = 0
	for i in range(len(Vec)):
		sum = sum + np.square(Vec[i])
	return np.sqrt(sum)

def Euclidean_Distance(Vec1, Vec2):
	return np.linalg.norm(Vec1 - Vec2)

def output(condition, dic):
	print(condition)
	print("DocID\tScore")
	for file, score in dic[:5]:
		print("{}\t{}".format(file, round(score, 6)))

mypath = "documents"
files = listdir(mypath)
bloblist = [] #save for documents 
stop_words = set(stopwords.words('english'))
ps = PorterStemmer() #stemmer

#read queries and turn into vector and tf_normalize step 2.
query = {}
given_q = input("Please enter the queries :")
given_q = given_q.split(' ')
for word in given_q:
	word = ps.stem(word)
	if word not in query.keys():
			query[word] = 1
	else:
		query[word] = query[word] + 1

# read file step 1.
for f in files:
	after_work = {}
	fullpath = join(mypath, f)
	k = open(fullpath)
	iter_kk = iter(k);
	str = ""
	for line in iter_kk:
		str = str + line
	str = str.lower()
	str = remove_punctuation(str)
	str = remove_apostrophe(str)
	#remove stopwords & stemming & indexing
	word_tokens = word_tokenize(str) 
	filtered_sentence = [w for w in word_tokens if not w in stop_words] 
	for word in filtered_sentence:
		word = ps.stem(word)
		if word not in after_work.keys():
			after_work[word] = 1
		else:
			after_work[word] = after_work[word] + 1
	bloblist.append(after_work)

#documents tf_normalize step 3.
document_tf_list = [] #use for tf calculate
document_tfidf_list = [] #use for tfidf calculate
for blob in bloblist:
	d_sum = sum(blob.values())
	tf_scores = {key: tf_calculate(key, blob, d_sum) for key in blob.keys()}
	tfidf_scores = {key: tfidf(key, blob, bloblist) for key in blob.keys()}
	document_tf_list.append(tf_scores)
	document_tfidf_list.append(tfidf_scores)

#Term Frequency (TF) Weighting + Cosine Similarity
tf_c = {}
for i, document in enumerate(document_tf_list):
	size = len(document) + len(query)
	d = np.zeros([size])
	q = np.zeros([size])
	dif = 0
	for j, key in enumerate(document):
		d[j] = document[key]
		if(key in query.keys()):
			q[j] = tf(key, query)
	for j, key in enumerate(query):
		if(key not in document.keys()):
			q[len(document)+dif] = tf(key, query)
			dif += 1
	f = files[i].split('.')
	tf_c[f[0]] = Cosine_Similarity(q,d)
tf_c = sorted(tf_c.items(), key=lambda x: x[1], reverse=True)
output("Term Frequency Weighting + Cosine Similarity:", tf_c)

#Term Frequency (TF) Weighting + Euclidean Distance
tf_e = {}
for i, document in enumerate(document_tf_list):
	size = len(document) + len(query)
	d = np.zeros([size])
	q = np.zeros([size])
	dif = 0
	for j, key in enumerate(document):
		d[j] = document[key]
		if(key in query.keys()):
			q[j] = tf(key, query)
	for j, key in enumerate(query):
		if(key not in document.keys()):
			q[len(document)+dif] = tf(key, query)
			dif += 1
	f = files[i].split('.')
	tf_e[f[0]] = Euclidean_Distance(q, d)
tf_e = sorted(tf_e.items(), key=lambda x: x[1], reverse=False)
output("Term Frequency Weighting + Euclidean Distance:", tf_e)

#TF-IDF Weighting + Cosine Similarity
tfidf_c = {}
for i, document in enumerate(document_tfidf_list):
	size = len(document) + len(query)
	d = np.zeros([size])
	q = np.zeros([size])
	dif = 0
	for j, key in enumerate(document):
		d[j] = document[key]
		if(key in query.keys()):
			q[j] = tfidf(key, query, bloblist)
	for j, key in enumerate(query):
		if(key not in document.keys()):
			q[len(document)+dif] = tfidf(key, query, bloblist)
			dif += 1
	f = files[i].split('.')
	tfidf_c[f[0]] = Cosine_Similarity(q,d)
tfidf_c = sorted(tfidf_c.items(), key=lambda x: x[1], reverse=True)
output("TF-IDF Weighting + Cosine Similarity:", tfidf_c)

#TF-IDF Weighting + Euclidean Distance
tfidf_e = {}
for i, document in enumerate(document_tfidf_list):
	size = len(document) + len(query)
	d = np.zeros([size])
	q = np.zeros([size])
	dif = 0
	for j, key in enumerate(document):
		d[j] = document[key]
		if(key in query.keys()):
			q[j] = tfidf(key, query, bloblist)
	for j, key in enumerate(query):
		if(key not in document.keys()):
			q[len(document)+dif] = tfidf(key, query, bloblist)
			dif += 1
	f = files[i].split('.')
	tfidf_e[f[0]] = Euclidean_Distance(q, d)
tfidf_e = sorted(tfidf_e.items(), key=lambda x: x[1], reverse=False)
output("TF-IDF Weighting + Euclidean Distance:", tfidf_e)

#Feedback Queries + TF-IDF Weighting + Cosine Similarity
ftfidf_c = {}
feedback_query = []
feedback = {}
for i, document in enumerate(bloblist):
	f = files[i].split('.')
	
	if (f[0] == tfidf_c[0][0]):
		temp = []
		for key in document:
			temp.append(key)
		
		tagged_words = nltk.pos_tag(temp)
		for key, apple in tagged_words:
			if(apple[:2] == "NN" or apple[:2] == "VB"):
				feedback_query.append(key) 
		
for word in feedback_query:
	if word not in feedback.keys():
		feedback[word] = 1
	else:
		feedback[word] = feedback[word] + 1

for key in feedback:
	feedback[key] = tfidf(key, feedback, bloblist) * 0.5

for i, document in enumerate(document_tfidf_list):
	size = len(document) + len(query) + len(feedback_query)
	d = np.zeros([size])
	q = np.zeros([size])
	dif = 0
	for j, key in enumerate(document):
		d[j] = document[key]
		if(key in query.keys()):
			q[j] = q[j] + tfidf(key, query, bloblist)
		if(key in feedback.keys()):
			q[j] = q[j] + feedback[key]
	for j, key in enumerate(query):
		if(key not in document.keys()):
			if(key in feedback.keys()):
				q[len(document)+dif] = tfidf(key, query, bloblist) + feedback[key]
			else:
				q[len(document)+dif] = tfidf(key, query, bloblist)
			dif += 1
	for j, key in enumerate(feedback):
		if(key not in document.keys() and key not in query.keys()):
			q[len(document)+dif] = feedback[key]
			dif += 1
	f = files[i].split('.')
	ftfidf_c[f[0]] = round(Cosine_Similarity(q,d), 6)
ftfidf_c = sorted(ftfidf_c.items(), key=lambda x: x[1], reverse=True)
output("Feedback Queries + TF-IDF Weighting + Cosine Similarity:", ftfidf_c)
