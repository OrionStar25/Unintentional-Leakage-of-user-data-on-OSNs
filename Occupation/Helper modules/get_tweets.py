# encoding: utf-8

import json
import tweepy
import re
from credentials import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

#-------------------------------------------------------------------------------------------------------#

def remove_url(text_msg):
	return re.sub(r"http\S+", "", text_msg)

def clean_tweets(text):
	#remove urls
	text = remove_url(text)

	#remove punctuations
	text = re.sub(r'[^\w\s]','',text)

	#remove stopwords
	cach_sw = stopwords.words("english")
	text = ' '.join([word.lower() for word in text.split() if word not in cach_sw])
	return text
#-------------------------------------------------------------------------------------------------------#

global cach_sw
cach_sw = stopwords.words("english")

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

file1 = open("user-tweettexts","w") 

# read job-tweetids ka 2nd column
with open('jobs-tweetids','r') as f:
	for line in f:
		user_id,tweet_id = line.split()
		try:
			tweet = api.get_status(tweet_id)
			if tweet.lang=="en":
				clean_tweet = clean_tweets(tweet.text)
				file1.write(user_id + "\t" + clean_tweet + "\n") 
		except:
			print "Invalid user"

file1.close()