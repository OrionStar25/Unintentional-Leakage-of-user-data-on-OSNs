import tweepy
import numpy as np
import json
import pickle
import csv
from time import time
import pandas as pd

auth = tweepy.OAuthHandler('e8JaecuxIXomoQffVi7ja3lVS', 'avdFCjcVVcami5EEEygWrT7vuKqkjJtc5OhoBn1L57ALA7UlRG')
auth.set_access_token('715443893442101248-sgAs7czVPj4SxMWvJjLNs5KBNDH2xT5', 'XBLDRALBOWgJETFxtchsQILifToRcNj9zS604pcoJjs7s')
api = tweepy.API(auth)

male_female_tag_data = pd.read_csv("final_data.csv", encoding='latin1')

print(len(male_female_tag_data))

def get_all_tweets(final_data): 
	i=0
	for index, row in final_data.iterrows():
		i+=1
		try:
			print(i)
			print(row['name'])
			new_tweets = api.user_timeline(screen_name=row['name'], count=50, trim_user=True,)

			outtweets = [[u''.join(row['name']).encode('utf-8'), u''.join(row['gender']).encode('utf-8'), u''.join(tweet.text).encode('utf-8')] for tweet in new_tweets]

			with open('extracted_tweets.csv', 'a') as f:
				writer = csv.writer(f)
				writer.writerows(outtweets)
				f.close()
				print('saved {} tweets for user {} to csv'.format(len(new_tweets), row['name']))
		except:
			print('passing..' + " " + row['name'])
			pass

alltweets = get_all_tweets(male_female_tag_data)