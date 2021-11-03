import requests
import datetime
from ipynb.fs.full.keys import *

import tweepy

consumer_key = twit_key
consumer_secret = twit_secr
access_token = twit_token

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)



# get tweets from certain dates?
startDate = datetime.datetime(2021, 5, 24, 0, 0, 0)
endDate =   datetime.datetime(2021, 5, 24, 0, 0, 0)


len(tweets)

ttweets = []
for tweet in tweets:
    if tweet.created_at < endDate and tweet.created_at > startDate:
        ttweets.append(tweet)



# get user
snow = api.get_user(screen_name = 'Snowden')
snow_id = snow.id
tweets = api.user_timeline('Snowden', since=startDate, until=endDate, count=500)
first_tweet = tweets[0]

ttweets = []
for tweet in tweets:
    if tweet.created_at < endDate and tweet.created_at > startDate:
        ttweets.append(tweet)

len(ttweets)

print(first_tweet)

