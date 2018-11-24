'''
INTEREST PREDICTION ( SUB PART OF MINI PROJECT )
APPROACH - NAIVE BAYES , TF - IDF , K-BEST FEATURES 
'''

#importing pandas
import pandas

#importing for writing in csv format
import csv,re

#import for generaion, twitter auth and time for analysis.
import random,tweepy,datetime

#importing matplotlib
import matplotlib.pyplot as plt

#importing nltk library.
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

#importing th sklearn library for classifiers
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#importing libraries for accuracy and testing purpose
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import KFold, cross_val_score

#import for joblib
from sklearn.externals import joblib

#Importing Counter from collections.
from collections import Counter

#----------------------------------------------------------------------------------------------------

'''
Testing dataset
( can be increased )
'''
testing_data_vector = pandas.read_csv('test_file.csv')

#----------------------------------------------------------------------------------------------------

def generate_the_user_array():

	dum = {}

	dum['news'] = ['cnnbrk', 'nytimes', 'ReutersLive', 'BBCBreaking', 'BreakingNews']
	dum['inspiration'] = ['DalaiLama', 'BrendonBurchard', 'mamagena', 'marcandangel', 'LamaSuryaDas']
	dum['sports'] = ['espn', 'SportsCenter', 'NBA', 'foxsoccer', 'NFL']
	dum['music'] = ['thedailyswarm','brooklynvegan','atlantamusic','gorillavsbear','idolator']
	dum['fashion'] = ['bof','fashionista_com','glitterguide','twopointohLA','whowhatwear']
	dum['gaming'] = ['IGN','Kotaku','Polygon','shacknews','gamespot']
	dum['politics'] = ['potus','ezraklein','politicalwire','nprpolitics','senatus']
	dum['tech'] = ['TheNextWeb','recode','TechCrunch','TechRepublic','Gigaom']
	dum['finance'] = ['jimcramer', 'pimco','StockTwits','stlouisfed','markflowchatter']
	dum['food'] = ['nytfood','Foodimentary','TestKitchen','seriouseats','epicurious']

	return dum
#------------------------------------------------------------------------------------------------------
#Function for extracting the tweet from the tweeter site.
def get_the_tweets(dataset):

	# if no function specified, read csv from file
	if dataset == None:

		print("******--Tweets are getting loaded from - ex_tw.csv--******")
		#reading from csv file.
		return pandas.read_csv('ex_tw.csv')

	# tweet extraction using official twitter API.
	else:

		#Open CSV file
		with open('ex_tw.csv','w') as tweeter_dataset:

			w = csv.writer(tweeter_dataset)

			w.writerow(['name','interest','text'])

			#Getting API access of twitter
			API = Tweeter_Auth_Fnx(random.randrange(0,5))
		
			for id_str, item in dataset.items():

				print('Topic: '+ str(id_str))
				for i in range(len(item)):

					done = 0
					print('User name : ' + item[i] + ' consumed time ' + str(datetime.datetime.now() - starttime)) 

					while True:
						if done > 0:
							break

						try:               
							tweet_list = []
							session = API.user_timeline( screen_name = item[i] , count = 200 )

							tweet_list.extend(session)
							last_one = tweet_list[-1].id - 1

							#No of tweets = 3200
							counter = 16
							
							while len(session) > 0 and counter > 1:

								counter -= 1

								session = API.user_timeline( screen_name = item[i] , count = 200 , max_id = last_one )
								tweet_list.extend(session)

								last_one = tweet_list[-1].id - 1

							tweet_list = [rmv_url(tweet.text) for tweet in tweet_list]
								
							for tw in tweet_list:
								w.writerow([u''.join(item[i]).encode('utf-8'),u''.join(id_str).encode('utf-8'),u''.join(tw).encode('utf-8')])
							
							done += 1

						except:

							done = 0
							continue

		return pandas.read_csv('ex_tw.csv')
#----------------------------------------------------------------------------------------------------------------

# takes in a pandas df_vector and returns a feature vector (sparse matrix) and list of interests which act as labels
def get_vector_and_labels(df_vector,feature_Selector):

	global tf_idf_tr, selector

	print("***** DATASET ---> VECTOR !!! *******")

	tweet_list, label_As_Interest = convert_to_tweets_interest(df_vector)

	print("****** Stemming GOING --> --> -->!! ******")

	tweet_list = stemming_fnx(porter,tweet_list) # change stemming algorithm here
	data_counts = cnt_vect.fit_transform(tweet_list)

	if feature_Selector:

		temp_selector = selector.fit(data_counts,label_As_Interest)
		data_counts = temp_selector.transform(data_counts)
		# data_counts = selector.fit(data_counts, label_As_Interest).transform(data_counts)

	print("******** %s features ********" % data_counts.shape[1])

	temp_tf_idf_tr = TfidfTransformer(use_idf=True).fit(data_counts)

	tf_idf_tr = temp_tf_idf_tr
	tf_idf_array = TfidfTransformer(use_idf=True).fit_transform(data_counts)

	return tf_idf_array, label_As_Interest

#------------------------------------------------------------------------------------------------------

# takes in a stemmer object defined in initalizeCV() and a list of strings to be stemmed
def stemming_fnx(stemmer, tweet_list):

	if stemmer != None:

		st_tokn =[]

		for phrase in tweet_list:

			tokn = phrase.split(' ')
			tokn = [stemmer.stem(tok) for tok in tokn if not tok.isdigit()]
			st_tokn.append(tokn)

		tweet_list = []

		for tokn in st_tokn:
			tweet_list.append(" ".join(str(i) for i in tokn))

	return tweet_list

#------------------------------------------------------------------------------------------------------

#Cleaning of data_Set ;; Removing punctuations , stopwords.
def convert_to_tweets_interest(df_vector):

	tweet_list = []
	label_As_Interest = []

	for i in df_vector.index:

		twt = df_vector.text[i]
		intst = df_vector.interest[i]

		if type(twt) == str:

			# remove punctuation
			twt = re.sub(r'[^\w\s]','',twt)

			# remove stopwords and change to lowercase
			twt = ' '.join([word.lower() for word in twt.split() if word not in cach_sw])

			tweet_list.append(twt)
			label_As_Interest.append(intst)

	return tweet_list, label_As_Interest

#-----------------------------------------------------------------------------------------------------

# train multinomial naive bayes classifier with given features and labels
def Naive_Bayes_Training(f,l):
	clfier = MultinomialNB().fit(f, l)
	return clfier

#-----------------------------------------------------------------------------------------------------

def fnx_answer(handle,classifier,tweet_count,feature_Selector):

	tweet_list = tweet_miner(handle,tweet_count)

	data_counts = cnt_vect.transform(tweet_list)

	if feature_Selector:

		temp_selector = selector
		data_counts = temp_selector.transform(data_counts)

	tf_idf_array = TfidfTransformer(use_idf=True).fit(data_counts).transform(data_counts)

	pred_array  = classifier.predict(tf_idf_array)
	phr = (word for word in pred_array  if word[:1])

	#return the answer!!!
	answer = Counter(phr).most_common(1)[0][0] 
	return answer

#------------------------------------------------------------------------------------------------------

def tweet_miner(handle,tweet_count):

	API = Tweeter_Auth_Fnx(3)
	tweet_list = []

	no_of_tweet = tweet_count // 200 # max number of tweets per request is 200

	print('**** Extracting  %s tweets of %s' % ((no_of_tweet)*200, handle))
	session = API.user_timeline(screen_name = handle,count=200)
	
	tweet_list.extend(session)
	last_one = tweet_list[-1].id - 1

	while len(session) > 0 and no_of_tweet > 1:

		no_of_tweet -= 1

		session = API.user_timeline(screen_name = handle, count=200 ,  max_id = last_one)

		tweet_list.extend(session)
		last_one = tweet_list[-1].id - 1

	tweet_list = [rmv_url(i.text) for i in tweet_list]

	return tweet_list

#-----------------------------------------------------------------------------------------------------------

def rmv_url(text_msg):
	return re.sub(r'^https?:\/\/.*[\r\n]*', '', text_msg)

#-----------------------------------------------------------------------------------------------------------

# uses the cross_val_score method to calculate the accuracy of a model using kfold cross validation, with cv being the number of folds
def K_FOLD_func(cfr, f, l, name):	
	kfold_score = cross_val_score(cfr, f , l , cv=10)
	print("Accuracy : " + str(kfold_score.mean()))

#------------------------------------------------------------------------------------------------------------

def Tweeter_Auth_Fnx(keyNum):

	auth = tweepy.OAuthHandler('e8JaecuxIXomoQffVi7ja3lVS', 'avdFCjcVVcami5EEEygWrT7vuKqkjJtc5OhoBn1L57ALA7UlRG')
	auth.set_access_token('715443893442101248-sgAs7czVPj4SxMWvJjLNs5KBNDH2xT5', 'XBLDRALBOWgJETFxtchsQILifToRcNj9zS604pcoJjs7s')
	return tweepy.API(auth)

#------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

	#begin variable is used as false/true true denotes collecting the dataset of the user/false , already collected.
	begin = False

    #Generate the array of 50 users
	#data_set = generate_the_user_array()

   	df_vector = get_the_tweets(dataset = None)

	print("******* Settup getting ready!! **********")
	
	#This is used directly from internet for better training purpose
	global cnt_vect, cach_sw, porter, selector

	cnt_vect = CountVectorizer(ngram_range=(1,3)) #change settings for unigram, bigram, trigram
	porter = PorterStemmer()
	cach_sw = stopwords.words("english")
	selector = SelectKBest(f_classif, 36000) # select top k features 

	tf_idf_array, label_As_Interest = get_vector_and_labels(df_vector,feature_Selector = True) # true: feature selection ON

	if begin:
		# joblib is an sklearn library that allows us to save / load the trained classifiers
		NB_CFR = joblib.load('clf/nb.pkl')
		print("***** Naive Bayes Classifier loaded ******")
	else:

		NB_CFR = Naive_Bayes_Training(tf_idf_array,label_As_Interest)
		joblib.dump(NB_CFR,'cfr/nb.pkl')

	K_FOLD_func(NB_CFR,tf_idf_array,label_As_Interest,"NBClassifier")

	FLAG = True
	classifiers = NB_CFR

	while FLAG:

		print("***** -Press Ctrl+Z to exit- *******")
		handle = input("*** Input an existing twitter handle :) :  ")
		answer = fnx_answer(str(handle),classifiers,400,feature_Selector = True)
		print(handle + " -----> " + answer)

#-------------------------------------------------------------------------------------------------------------------
