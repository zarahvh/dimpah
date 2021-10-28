#  Hate Speech Detector

In today's session, we learned that in order to detect sentiments we can simply compare freqeuencies of positive and negative words. To this end, we downloaded a dictionary of such terms from the web and then determined their respective frequency. If there are more positive terms in a document than negative ones, we considered it to have a positive sentiment and otherwise a negative one.

There are many such dictinonaries produced by linguists but also other communities such as journalists. We can use these with the same approach we used for detecting sentiments to understand texts in different contexts. Journalists, for instance, have developed https://www.hatebase.org/, the world's largest online repository of structured, multilingual, usage-based hate speech. 

Here, we will use hatebase to develop a hate speech detector for tweets by counting the number of hate words in tweets. We will concentrate on the English language. You can go to https://www.hatebase.org/ and explore the search functions to take a look at the English terms in hatebase. 



Next, we need to download the hatebase dictionary, which is unfortunately not that easy. You need to register for an API key and then work relatively hard to get the API to return all English hate speech terms. 

I have commented out the hate_vocabulary(api_key) function that speaks to https://www.hatebase.org/ and instead provided you with a direct import from a local CSV file. If you want to, for instance, download the dictionary for another language than English, you need to un-commnent those lines.

import pandas as pd

hate_df = pd.read_csv('https://raw.githubusercontent.com/goto4711/social-cultural-analytics/master/hate-vocab-eng.csv')

hate_df.head()

Next we access Twitter the way we learned today. The code is set to a query Twitter about 'Trump' below but is not active. In order to activate it, you need to add your Twitter API details. You can of course also change the search_term.

import tweepy
import requests
from ipynb.fs.full.keys import *

consumer_key = twit_key
consumer_secret = twit_secr
access_token = twit_token

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

search_term = 'Trump'
tweets = api.search(q=search_term)

for tweet in tweets:
    print(tweet.text)

Rather than performing a live Twitter search, I have saved the a search on 'Trump' from the day of his 2017 inauguration. That's the read.csv command further down. Please, note that the file was created with the old twitteR library, which means that some of the column names are different from what you are used to. But the one we are interested in is still called 'text'.

tweets = pd.read_csv("https://raw.githubusercontent.com/goto4711/social-cultural-analytics/master/trump-tweets-20-1.csv", encoding='latin-1')

Let's take a look at tweets. You will see all texts as well as a lot of other information.

tweets.head()

Next we start with our standard text mining workflow 

Unfortunately, tweets can be difficult to process, as people use very different types of language, of formatting, etc. I have therefore provided you with a clean_tweets function, which applies to all the texts in the tweets and save the results in a tweet_list

def clean_tweets(df):
    text = df['text']
    tweet_list = []
    for tweet in text:
        tweet = tweet.split()
        tweet = ["" if len(word) < 3 else word for word in tweet]
        tweet_list.append(tweet)
    return tweet_list
    
tweet_list = clean_tweets(tweets)
tweet_list_new = []
for tweet in tweet_list:
    tweet_str = " ".join(tweet)
    tweet_list_new.append(tweet_str)


Next, our ususal steps to prepare a TM corpus.

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopwords = list(stopwords.words('english'))

def Corpuser(corpus):
    corpus = word_tokenize(corpus)
    corpus = [word.replace(" ", "") for word in corpus]
    corpus = [word.lower() for word in corpus if word.isalpha()]

    corpus = [word for word in corpus if word not in stopwords]
    
    return corpus

# tweet_corp = Corpuser(tweet_list_new)
# print(tweet_corp)

docs = []
for tweet in tweet_list_new:
    doc = Corpuser(tweet)
    docs.append(str(doc))


Our next TM workflow step will be to create a term-document-matrix to count the terms in the documents.

from nltk import *

tf = FreqDist(docs)
print(tf)

from sklearn.feature_extraction.text import CountVectorizer 

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
dtm = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
dtm = dtm.T
dtm

All we have to do now is find out which rownames (terms) of tdm correspond to terms in our hate speech dictionary. The columns (docs) of tdm that are larger than 0 are then the tweets which contain hate speech words.

The python function isin answers the question: 'Where do the values in the hate vocabulary appear in the dataframe'

hate_voc = hate_dict['word'].values.tolist()
hate_voc = [word.lower() for word in hate_voc if word.isalpha()]

hate_speech = dtm[dtm.index.isin(hate_voc)]

hate_speech

Now we only need to find the indexes of these words to see to which tweet they belong. The columns (docs) of tdm that are larger than 0 are then the tweets which contain hate speech words.

hate_speecht = hate_speech.T
bitch = hate_speecht.index[hate_speecht['bitch'] > 0].to_list()
idiot = hate_speecht.index[hate_speecht['idiot'] > 0].to_list()
print(bitch)
print(idiot)

Let's check out the tweets that contain 'bitch

tweets_bitch = tweets.iloc[bitch]['text']
for tweet in tweets_bitch:
    print(tweet)

Some of these are very angry about Trump, but probably still not really hate speech. This shows the limitations of the approach to use simple words and phrases.

But this approach can still be useful to filter tweets for manual review by editors. Twitters and others actually have engines like this. It is frequently used in apps like http://www.huffingtonpost.com/entry/donald-trump-stock-alert_us_586e67dce4b0c4be0af325fc, which sends alerts when Donald Trump tweets about your stocks.