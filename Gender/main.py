# coding: utf-8

import pandas, csv, re, datetime, tweepy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2


Count_Vectorizer = CountVectorizer(ngram_range=(1,3))
cachedStopwords = stopwords.words("english")
selector = SelectKBest(f_classif, 50000)

def authTwitter(keyNum):
	auth = tweepy.OAuthHandler('e8JaecuxIXomoQffVi7ja3lVS', 'avdFCjcVVcami5EEEygWrT7vuKqkjJtc5OhoBn1L57ALA7UlRG')
	auth.set_access_token('715443893442101248-sgAs7czVPj4SxMWvJjLNs5KBNDH2xT5', 'XBLDRALBOWgJETFxtchsQILifToRcNj9zS604pcoJjs7s')
	return tweepy.API(auth)


# calls the twitter API to return a number of tweets from a specified twitter handle
def mineTweets(targetHandle):
	api = authTwitter(3)
	numOfTweets = 100
	listOfTweets = []
	print('Mining %s tweets from %s' % (100, targetHandle))
	try:
		outtweets = api.user_timeline(screen_name = targetHandle,count=100)
		listOfTweets = [tweet.text for tweet in outtweets]
		#listOfTweets = [removeUrl(tweet.text) for tweet in listOfTweets]
	except:
		print('passing...')
	return listOfTweets


def getFeatureVectorAndLabels(dataframe):
	print("getting tweet and labels lists")
	tweetList, labels = getListOfTweetAndLabels(dataframe)
	print(len(tweetList))
	print(len(labels))

	cv = Count_Vectorizer.fit_transform(tweetList)
	temp_selector = selector.fit(cv,labels)
	cv = temp_selector.transform(cv)

	tfidf_final = TfidfTransformer(use_idf=True).fit_transform(cv)
	return tfidf_final, labels


def getListOfTweetAndLabels(dataframe):
	tweetList = []
	labels = []
	for i in dataframe.index:
		tweet = dataframe.text[i]
		label = dataframe.gender[i]
		if type(tweet) == str:
			# remove punctuation
			tweet = re.sub(r'[^\w\s]','',tweet)
			# remove stopwords and change to lowercase
			tweet = ' '.join([word.lower() for word in tweet.split() if word not in cachedStopwords])
			tweetList.append(tweet)
			labels.append(label)
	return tweetList, labels


def train_classifier(dataframe):
	tfidf_final, labels = getFeatureVectorAndLabels(dataframe)
	print("Starting to train the classifier")
	starttime = datetime.datetime.now()
	classifier = MultinomialNB().fit(tfidf_final, labels)
	print("Time taken to train NBClassifier: " + str(datetime.datetime.now() - starttime))

	printKFoldScore(classifier,tfidf_final,labels,"NBClassifier")

	return classifier


def get_gender(targetHandle, classifier):
	listOfTweets = mineTweets(targetHandle)
	#print(listOfTweets)
	if len(listOfTweets) > 0:
		data_counts = Count_Vectorizer.transform(listOfTweets)
		temp_selector = selector
		data_counts = temp_selector.transform(data_counts)
		tfidf_doc = TfidfTransformer(use_idf=True).fit(data_counts).transform(data_counts)
		predictedList = classifier.predict(tfidf_doc)
		numWords = (word for word in predictedList if word[:1])
		targetGender = Counter(numWords).most_common(1)[0][0]

		return targetGender
	else:
		return "none"


# uses the cross_val_score method to calculate the accuracy of a model using kfold cross validation, with cv being the number of folds
def printKFoldScore(classifier, features, labels, name):	
	kfold_score = cross_val_score(classifier, features, labels, cv=10)
	print("Accuracy for " + name +  ": " + str(kfold_score.mean()))


def main():
	dataframe = pandas.read_csv('extracted_tweets.csv', low_memory = False)
	classifier = train_classifier(dataframe)
	joblib.dump(classifier,'classifiers/naivebayes.pkl')
	#classifier = joblib.load('classifiers/naivebayes.pkl')
	'''
	while True:
		print("Press Ctrl+D to exit")
		targetHandle = input("Please enter a valid twitter handle: ")
		targetInterest_NB = get_gender(str(targetHandle),classifier)
		print(targetHandle + " => " + targetInterest_NB + " (NBClassifier)")
	'''

	test_data = pandas.read_csv('test_data.csv', low_memory = False)
	correct = 0
	wrong = 0
	for i in test_data.index:
		targetHandle = test_data.name[i]
		targetInterest_NB = get_gender(str(targetHandle), classifier)
		if targetInterest_NB == test_data.gender[i]:
			correct += 1
		elif targetInterest_NB != "none": 
			wrong += 1
		print(str(correct) + " " + str(wrong))  
		print(targetInterest_NB + " " + test_data.gender[i])


if __name__ == '__main__':
  main()
