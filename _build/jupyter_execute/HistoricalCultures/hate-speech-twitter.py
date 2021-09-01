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
# tweets = tweepy.Cursor(api.search, q = search_term)
tweets = api.search(q=search_term)

for tweet in tweets:
    print(tweet.text)

Rather than performing a live Twitter search, I have saved the a search on 'Trump' from the day of his 2017 inauguration. That's the read.csv command further down. Please, note that the file was created with the old twitteR library, which means that some of the column names are different from what you are used to. But the one we are interested in is still called 'text'.

tweets = pd.read_csv("https://raw.githubusercontent.com/goto4711/social-cultural-analytics/master/trump-tweets-20-1.csv", encoding='latin-1')
                

Let's take a look at tweets. You will see all texts as well as a lot of other information.

tweets.head()

Next we start with our standard text mining workflow 

Unfortunately, tweets can be difficult to process, as people use very different types of language, of formatting, etc. I have therefore provided you with a clean_tweets function, which applies to all the texts in the tweets and save the results in a tweets_text list.

def clean_tweets(df):
    text = df['text']
    tweet_list = []
    for tweet in text:
#         print(tweet)
        tweet = tweet.split()
        for word in tweet:
            if len(word) < 3:
                word.replace(word, "")
        tweet_list.append(tweet)
    return tweet_list
    
tweet_list = clean_tweets(tweets)
tweet_list_str = str([tweet for tweets in tweet_list for tweet in tweets])

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

Corpuser(tweet_list_str)

Our next TM workflow step will be to create a term-document-matrix to count the terms in the documents. You might have noticed above that the hatebase vocabulary contains not just single words but also phrases of more than one word such as 'African catfish'. As we also learned today, these are so-called bigrams (2 word phrases). So, we create two term-document-matrices one for the single words (also called unigrams) and one for the bigrams.



# first we create a frequency table

def frequencytable(corpus):
    words = Corpuser(corpus)
    freq_table = {}
    for word in words:
        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1
    return freq_table

table = frequencytable(tweet_list_str)

# then create the actual dataframe where each tweet is a column
# possible for a few tweets but might be too complex for 1000?
# is it needed to distinguish every tweet or is it about the entire corpus? --> Yes you want to know 
# how many tweets contain hatespeech

dfs = []
for i in range(1000):
    table = frequencytable(str(tweet_list[i]))
    i = pd.DataFrame.from_dict(table, orient='index', columns={i})
    dfs.append(i)

merged_df = pd.concat(dfs, axis=1)
merged_df = merged_df.fillna(0)

merged_df

# now we check which of the tweets contain hate speech thus therefore match withe the hate speech dictionary
# --> but where are the hate speech words? is it about the offensivenes?
# what exactly is being done with the bigram?

# hate_dff = hate_df.loc[hate_df['offensiveness'] > 0]

hate_dict = hate_df[['word', 'offensiveness']]

hate_dict

All we have to do now is find out which rownames (terms) of tdm correspond to terms in our hate speech dictionary. 

# we drop the rownames that are not in the hate_dict

hate_voc = hate_dict['word'].values.tolist()
hate_voc = [word.lower() for word in hate_voc if word.isalpha()]

hate_speech = merged_df[merged_df.index.isin(hate_voc)]

The columns (docs) of tdm that are larger than 0 are then the tweets which contain hate speech words.



# hate = hate_speech[hate_speech > 0]

hate_voc

# what is the hate speech vocab??
hate_speech

merged_df

exists = 'trump' in tweets.text
print(exists)

merged_df.index.isin(hate_voc)

merged_df.index

words = word_tokenize(tweet_list_str)
words = [word.replace(" ", "") for word in words]
words = [word.lower() for word in words if word.isalpha()]

exists = 'trump' in words
print(exists)

words

# Why doesn't it find the words that are there? such as bitch, bubble, etc....