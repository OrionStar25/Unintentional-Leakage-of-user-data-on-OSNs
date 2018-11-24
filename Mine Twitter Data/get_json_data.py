# encoding: utf-8

import json
import tweepy
from credentials import *



#-----------------------------------------------------------------------------------------#
# Get twitter data in JSON format

# import access tokes from credentials.py

# consumer_key = ""
# consumer_secret = ""
# access_token = ""
# access_token_secret = ""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth) 

#Put your search term
searchquery = "code"

users =tweepy.Cursor(api.search,q=searchquery).items()
count = 0
errorCount=0

file = open('output.json', 'wb') 
data=[]

while True:
    try:
    	user = next(users)
    	count += 1
        if (count>=10): # Can change number of tweets here
        	break
    except tweepy.TweepError:
        print "sleeping...."
        time.sleep(60)
        user = next(users)
    except StopIteration:
        break
    try:
        print "Writing to JSON tweet number:"+str(count)
        data.append(user._json)        
    except UnicodeEncodeError:
        errorCount += 1
        print "UnicodeEncodeError,errorCount ="+str(errorCount)



json.dump(data,file,sort_keys = True,indent = 4)
print "completed, errorCount ="+str(errorCount)+" total tweets="+str(count)


#-----------------------------------------------------------------------------------------#

