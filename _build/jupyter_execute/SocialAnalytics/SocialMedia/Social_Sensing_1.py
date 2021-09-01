# Social Sensing 1

There are a lot of social, cultural and other datasets out there on the web. We have already learned how to extract data from web sites directly. Many data items can also be accessed via a so-called API (Application Programming Interface). You can think of an API as a window through which you have access to remotely stored data rather than web pages. Often access to these APIs is limited by some kind of registration key you have to use to open that window. Before you proceed with this lesson, please register now for the two services we will work with today. The first one is the US data.gov site, which contains all kinds of datasets – mostly by the US government. You can register for a key under https://api.data.gov/signup. The second online service is Twitter and does not need an introduction. You can sign up for the several keys we need from Twitter at https://apps.twitter.com/. Here you can create an app that gives you the key and consumer secret.



## requests to call api in python
To call an api in python and access data from a website we need the requests package first, download and import it.


import requests

Now, we want to play with electricity rates and associated coordinate information for US locations by looking them up on data.gov. Let’s first define a place we are interested in. Because we want to use it again later, let’s assign address_ex = ‘1600 Amphitheatre Parkway, Mountain View, CA’. Do you know who ‘lives’ at this address?

from ipynb.fs.full.keys import *

address_x = '1600 Amphitheatre Parkway, Mountain View, CA'
api_key = gov_key
url = 'https://developer.nrel.gov/api/utility_rates/v3.json'

Requests has a get function that allows you to read/get data from remote sites.

params = dict(api_key=api_key, address=address_x)
req_1 = requests.get(url, params=params)

Finally, we can use another requests function to access the results or the actual text of the API call. Type in result = request.text

result = req_1.text
result

Unfortunately, result is a rather complex list. To get, for instance, the utility name, we have to first convert the response into json data so we can access it more easily.

request_data = req_1.json()
print(request_data)

We can then access parts of it by seeing it as a dictionary, to get the utitility name we need to first access outputs and then access utiltiy_name

request_data['outputs']['utility_name']

To get the residential electricity rate at the address, look for outputs and then residential.

request_data['outputs']['residential']

Google has many useful APIs. For instance, maps.googleapis.com returns the geo-locations for addresses. Again, access is restricted but you can run a few requests without a key. Let’s look up the geo-location of our address_ex, which is of course the location of the Googleplex, Google’s HQ. Again, get the location information

--> An API key is needed which costs money

params = dict(address = address_x, key = 'AIzaSyDQ8DkiK3SV9V2TpwekTwHmTEXYvuHNEP8')
url = 'https://maps.googleapis.com/maps/api/geocode/json?'
req_2 = requests.get(url=url, params=params)

req_2.text

--> It says I'm not autohorized to use this API


We are interested in the latitude and longitude of the Googleplex. Let's first convert it to readable data again



The longitude is available if we change 'lat' to 'lng'

You could now use http://www.latlong.net/Show-Latitude-Longitude.html to map these longitude and latitude and would find the Googleplex.

We have just worked through simple API requests that got us locations. More interesting will be to access social media applications like Twitter and Facebook. We can also get their data through APIs. Twitter is especially popular. 

In python there is a library that makes accessing Twitter data simple. Download and import tweepy.

import tweepy


consumer_key = twit_key
consumer_secret = twit_secr
access_token = twit_token

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

We are connected to Twitter and can run queries. Let’s get Barack Obama's timeline and his first tweet first

obama = api.get_user(screen_name = 'BarackObama')
ob_id = obama.id
tweets = api.user_timeline('BarackObama', count=100)
first_tweet = tweets[0]

Let's check the first tweet of the dataset by typing first_tweet.text

first_tweet.text

To see all the attributes of a tweet, simply type tweets[0], but for now we only want to check the date using first_tweet.created_at

first_tweet.created_at

Could we find a way to plot Obama his twitter activity?

dates = []
for tweet in tweets:
    date = tweet.created_at
    date = date.date()
    dates.append(date)
dates

from collections import Counter
import matplotlib.pyplot as plt

counts = Counter(dates)
plt.bar(counts.keys(), counts.values())

Let’s move on from Obama. In order to access my own Twitter favourites, please type in api.favorites('tobias_blanke')

favorites = api.favorites('tobias_blanke')
favorite = favorites[0] 
favorite.text

So far, so good. Of course, these requests were quite simple. So, let’s try something more complicated. We start with first a look at the retweet structures and then a typical content analysis in Twitter. Tweets are little pieces of texts with lots of metadata attached to them (https://en.wikipedia.org/wiki/Twitter). So, it is not surprising that many people try and run text analysis on the content of tweets. Let’s start with that.

The first step is to search Twitter for something of interest by running api.search(q=‘#kcl’, count = 10). This search will look for the most recent tweets (count=10) with the hashtag kcl.

kcl_tweets = api.search(q='#kcl', count =10)


As we can see each tweet has an id and a lot of other metadata attached to it such as retweets, locations, etc. Did you know that you produce so much information with each tweet?

There are a lot of things we can do with tweets and their metadata. Have a look at the documentation of the package or the many examples online. A quick example would be to return retweets. The first step is to find them. Looking at the metadata of each tweet in kcl_tweets, there are two relevant fields with retweet_count and retweeted. With retweet_count, we can check whether a tweet has been retweeted (retweet_count > 0), while retweeted tells us whether a tweet was a retweet itself. We want to find only those tweets that have not been retweets (retweeted == False) but are not a retweet. So, please select those kcl_tweets that have been retweeted. 

retweeted_kcl_tweets = []
for kcl_tweet in kcl_tweets:
    if kcl_tweet.retweeted == False and kcl_tweet.retweet_count > 0:
        retweeted_kcl_tweets.append(kcl_tweet)
        
retweeted_kcl_tweets   

As promised, we would like to run some simple content analysis with the text in the tweets. We will produce a simple word cloud. But before we can do this we need to first create a corpus from the tweets, the same as we did with the speeches in the Text assignment.
Let's first extract the text from the kcl_tweets.

tweet_text = []
for t in kcl_tweets:
    text = t.text
    tweet_text.append(text)


import nltk
from nltk.tokenize import word_tokenize

strings = ''.join(tweet_text)
corpus = word_tokenize(strings)
corpus = [word.replace(" ", "") for word in corpus]
corpus = [word.lower() for word in corpus if word.isalpha()]

from nltk.corpus import stopwords
stopwords = list(stopwords.words('english'))
corpus = [word for word in corpus if word not in stopwords]


from wordcloud import WordCloud


wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                min_font_size = 10).generate(str(corpus)) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

Hmmm https shows up there pretty big, let's check if everything went alright. 
Print the tweets that we just added to the corpus.

tweet = tweet_text[0]
print(tweet)

As we can see there is a link to the tweet in each text, we should have removed that first. We can use regular expressions for this.


import re
tweet = re.sub(r'http\S+', '', tweet)
print(tweet)

cleaned_tweets = []
for tweet in tweet_text:
    tweet = re.sub(r'http\S+', '', tweet) 
    cleaned_tweets.append(tweet)

cleaned_tweets
    

    

strings = ''.join(cleaned_tweets)
corpus = word_tokenize(strings)
corpus = [word.replace(" ", "") for word in corpus]
corpus = [word.lower() for word in corpus if word.isalpha()]

from nltk.corpus import stopwords
stopwords = list(stopwords.words('english'))
corpus = [word for word in corpus if word not in stopwords]

from wordcloud import WordCloud


wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                min_font_size = 10).generate(str(corpus)) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

Very popular with Twitter is also the analysis of followers. I don’t have so many. In fact, I am not really using Twitter much. But let’s still try. You can get my Twitter information by entering me = api.get_user(screen_name = 'tobias_blanke')

me = api.get_user(screen_name = 'tobias_blanke')
me.description

It’s me! My followers are a little bit more interesting. We can retrieve by going throug api.followers('tobias_blanke')

# get first 20 followers
for follower in api.followers('tobias_blanke'): 
    print(follower.screen_name)
    


I haven’t explained it yet, but Twitter limits the amount of API calls you can do at any moment in time, which is often an issue if you retrieve a lot of followers from accounts like Donalds Trump’s. Rate limits are often one of the biggest issues in Twitter analysis once the data gets a bit bigger. Check it out at https://dev.twitter.com/rest/public/rate-limiting. You will find plenty of people online who complain.

Which is also something that can happen while trying to access the information that we try to extract in this assignment, just in case that happens you can find a csv file with the same info in data.

Let's first get the ids of all the followers

# Get list of al the follower ids (https://towardsdatascience.com/how-to-download-and-visualize-your-twitter-network-f009dbbf107b)
tb_id = me.id
user = [tb_id]
follower_list = []
for user in user:
    followers = []
    try:
        for page in tweepy.Cursor(api.followers_ids, user_id=user).pages():
            followers.extend(page)
            print(len(followers))
    except tweepy.TweepError:
        print("error")
        continue
    follower_list.append(followers)

A key measure of my own importance on Twitter is the importance of the people who follow me. Does this make sense? Of course it does, as with important followers you can influence a lot of people. Let’s plot this measure and get an overview of the friends and followers of those who follow me. 
To do so we first want to create a dataframe that contains all my followers and their follower and friend count.

# Create dataframe
import pandas as pd

df = pd.DataFrame(columns=['user','follower'])
df['follower'] = follower_list[0]
df['user'] = tb_id

df

followers = follower_list[0]
fol_count = []
for follower in followers[:200]:
    try:
        user = api.get_user(follower)
        count = user.followers_count
        fol_count.append(count)
    except tweepy.TweepError:
        fol_count.append(0)
        print("error")
        continue
    

df200 = df.head(200)

df200['follower_count'] = fol_count

df200

friends_count = user.friends_count 

friends_count = []
for follower in followers:
    try:
        user = api.get_user(follower)
        count = user.friends_count
        friends_count.append(count)
    except tweepy.TweepError:
        friends_count.append(0)
        print("error")
        continue

df200['friends_count'] = friends_count

df200.to_csv('df200.csv')
df200

df200.plot.scatter(x='friends_count', y='follower_count',)

Ok, there are not too many strong performers in my followers’ list. In order to confirm this, let’s check the counts for all my followers with a plot. 

# Need to reload entire assignment first
# df200['follower_count'].count_values.plot()

So, most of my followers do not have too many followers themselves apart from one outlier. My influence is really limited. Let’s quickly move on then.

Social network analysis is really important both in social and cultural analytics. It uses graphs to explain and analyse social relations. We have already started talking about social networks. We looked into my followers and those friends that I am following. Then, we investigated the friends of these friends and the followers of these followers. To build these kinds of relationships and map them onto graphs to visualise and analyse them is really what social networks are all about.

We would like to build a graph of my friends and followers. We already have the followers, let's now reterive their screen names.

Because graph visualisation can quickly get confusing if there are too many items to represent, we would like to limit the number of friends and followers to 20.

screen_names = []
for follower in followers[:20]:
    try:
        user = api.get_user(follower)
        name = user.screen_name
        print(name)
        screen_names.append(name)
    except tweepy.TweepError:
        screen_names.append('error')
        print("error")
        continue

df20 = df200.head(20)
df20['screen_name_follower'] = screen_names
df20['screen_name_user'] = api.get_user(tb_id).screen_name
df20

# Check the friends of user/source
user = [tb_id]
n = 1
for user in user:
    friends = []
    screen_name_friends = []
    try:
        if n != 20:
            for friend in tweepy.Cursor(api.friends_ids, user_id=user).items():
                n+=1
                friends.append(friend)
                user = api.get_user(friend)
                name = user.screen_name
                screen_name_friends.append(name)
    except tweepy.TweepError:
        friends.append('error')
        print("error")
        continue


df_friends = pd.DataFrame(columns=['user','follower'])
df_friends['user'] = friends
df_friends['follower'] = tb_id


df_friends['screen_name_follower'] = api.get_user(tb_id).screen_name
df_friends['screen_name_user'] = screen_name_friends

df_friends['follower_count'] = len(followers)
df_friends['friends_count'] = len(friends)

df_friends.to_csv('df_friends.csv')
df_friends

merge = pd.concat([df20,df_friends])
merge

graphdata = merge

nodes = pd.DataFrame(columns=['userid', 'user_name'])
nodes['userid'] = graphdata.follower.unique()
nodes['user_name'] = graphdata.screen_name_follower.unique()

relations = pd.DataFrame(columns=['user', 'follower'])
relations['user'] = graphdata['user']
relations['follower'] = graphdata['follower']

nodes


import igraph
from igraph import *
graph = Graph(directed=True)

graph = Graph.DictList(
          vertices=nodes.to_dict('records'),
          edges=relations.to_dict('records'),
          directed=True,
          vertex_name_attr='userid',
          edge_foreign_keys=('follower', 'user'));

print(graph)


plot(graph, vertex_label=graph.vs['user_name'])
