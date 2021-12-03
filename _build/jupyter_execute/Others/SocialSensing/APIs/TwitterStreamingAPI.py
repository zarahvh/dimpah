# Twitter Streaming API

The Twitter Streaming API, one of 3 such APIs (search, streaming, “firehose“), gives developers (and data scientists!) access to multiple types of streams (public, user, site), with the difference that the streaming API collects data in real-time (as opposed to the search API, which retrieves past tweets).

import requests
import datetime
import tweepy
from ipynb.fs.full.keys import *


consumer_key = twit_key
consumer_secret = twit_secr
access_token = twit_token

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

class StreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)
    def on_error(self, status_code):
        if status_code == 420:
            return False

streamListener = StreamListener()
stream = tweepy.Stream(auth = api.auth, listener=streamListener)

import dataset
db = dataset.connect("sqlite:///tweets.db")

if coords is not None:
    coords = json.dumps(coords)
    
table = db["tweets"]
table.insert(dict(
    user_description=description,
    user_location=loc,
    coordinates=coords,
    text=text,
    user_name=name,
    user_created=user_created,
    user_followers=followers,
    id_str=id_str,
    created=created,
    retweet_count=retweets,
    user_bg_color=bg_color,
    polarity=sent.polarity,
    subjectivity=sent.subjectivity,))

stream.filter(track=['Putin'])

