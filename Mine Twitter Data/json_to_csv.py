import json
import sys
from csv import writer

with open("output.json") as in_file, \
     open("output.csv", 'w') as out_file:
    print >> out_file, 'tweet_id, tweet_author, tweet_author_id, tweet_language, tweet_geo, tweet_text'
    csv = writer(out_file)
    tweet_count = 0

    tweet_list = json.loads(in_file.read())
    for tweet in tweet_list:
        tweet_count += 1

        # Pull out various data from the tweets
        row = (
            tweet['id'],                    # tweet_id
            tweet['user']['screen_name'],   # tweet_author
            tweet['user']['id_str'],        # tweet_authod_id
            tweet['lang'],                  # tweet_language
            tweet['geo'],                   # tweet_geo
            tweet['text']                   # tweet_text
        )
        values = [(value.encode('utf8') if hasattr(value, 'encode') else value) for value in row]
        csv.writerow(values)
        
print "Done"