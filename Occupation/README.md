==Twitter Occupation Dataset==

Feature representation and tweets of a set of 5191 users mapped to their occupational class. Extracted around 5 August 2014.

Associated paper, read for more details:
Daniel Preotiuc-Pietro, Vasileios Lampos, Nikolaos Aletras
An analysis of the user occupational class through Twitter content
ACL 2015

Total number of users: 5191
Total number of tweet ids: 10796836

Contents:
1. jobs-tweetids - user_id[SPACE]tweet_id 	
   Each line represents a tweet.

2. jobs-unigrams - user_id[SPACE]wordid_1:frequency_1[SPACE]...wordid_n:frequency_n xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   Bag-of-words unigram feature representation, one user/line.

3. dictionary - wordid[SPACE]word
   Mapping between word ids and words.

4. jobs-users - user_id[SPACE]occupation_code xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   Resolved 3-digit SOC code for each user.

5. keywords - occupation_code,occupation_description,"keyphrase_1, ..., keyphrase_n"
   3-digit SOC code, its corresponding class description and the keyphrases for jobs in this category used for identifying users

If you are using this dataset, please cite:
@inproceedings{jobs15acl,
	title = {An analysis of the user occupational class through {T}witter content},
	journal = {Proceedings of the 53rd annual meeting of the Association for Computational Linguistics},
	year = {2015},
	series = {ACL},
	author = {Preo\c{t}iuc-Pietro, Daniel and Lampos, Vasileios and Aletras, Nikolaos}
}


Additional dataset created using the above data set:

1. user-tweettexts - user_id[SPACE]cleaned_tweet
2. labels - corresponding occupation_code

cleaned_tweet == valid tweets, i.e., that have not been deleted or whose user accounts are still valid/active