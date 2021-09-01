# Analysing Shakespeare

In this exercise, we will analyse Shakepeare using our new knowledge on text miming.

# download shakespeare

import requests

URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
page = requests.get(URL)

# print(len(page.text))

text = page.text

# the text can be split into several documents using \r\n\r\n\r\n in regex
# these newlines seperate each document

shakespeare = text.split("\r\n\r\n\r\n ")

# the second split is the first story
shakespeare[1]

# convert it to a dataframe
import pandas as pd

df = pd.DataFrame(shakespeare[1:155])
df

Finally, let's create a Shakespeare corpus and investigate the first 5 documents in it.

import numpy as np 

shakespeare_five = df.iloc[1:6]
shakespeare_five.describe()

Next, we apply our usual text transformations to clean the text, using the Corpuser function:

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

Create the DocumentTermMatrix:

# then we create a frequency table

docs = shakespeare[1:155]
def frequencytable(corpus):
    words = Corpuser(corpus)
    freq_table = {}
    for word in words:
        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1
    return freq_table

ft = frequencytable(str(docs))

def dtm(documents):
    dfs = []
    for i in range(len(documents)):
        table = frequencytable(str(documents[i]))
        i = pd.DataFrame.from_dict(table, orient='index', columns={i})
        dfs.append(i)
    dtm = pd.concat(dfs, axis=1)
    dtm = dtm.fillna(0)
    
    return dtm

shakespeare_dtm = dtm(docs)

shakespeare_dtm

Let's have a closer look at the first 10 docs

doc10 = shakespeare[1:11]
ft = frequencytable(str(doc10))
dtm_10 = dtm(doc10)

dtm_10

There are many many empty entries (0), which means the words do not appear in the document. We can for example use only the 10 most occuring terms

def top_n_terms(freqtab, n):
    sorted_ft = sorted(freqtab.items(), key=lambda x: x[1], reverse=True)
    freqtab = sorted_ft[:n]  
    terms = [tup[0] for tup in freqtab]
    return terms

# def dtm_10(documents):
#     dfs = []
#     for i in range(len(documents)):
#         table = frequencytable(str(documents[i]))
#         i = pd.DataFrame.from_dict(table, orient='index', columns={i})
#         dfs.append(i)
#     dfs
#     dtm = pd.concat(dfs, axis=1)
#     dtm = dtm.fillna(0)
    
#     return dtm

top10 = top_n_terms(ft, 10)
top10

dtm_top10 = dtm_10[dtm_10.index.isin(top10)]

dtm_top10

Better, much better ...


Let's find the 50 most frequent terms in Shakespeare next.

top50 = top_n_terms(ft, 50)
top50

## TF/IDF

We have discussed the tf/idf scoring for documents in the lecture. 

In python, tf/idf is easy to create using scikit learn. 

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
vec = tfidf.fit_transform(doc10)
 
matrix = pd.DataFrame(vec.toarray().transpose(), index=tfidf.get_feature_names())
 
matrix

## k-means clustering

Using, tf-idf we can apply our favourite clustering technique k-means to understand common clusters of words.

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10).fit(vec)

kmeans.labels_

Which word belongs to which cluser

# define columns
# why is there a 10 in there

kmeans_df = pd.DataFrame(kmeans.cluster_centers_, columns=matrix.index)
kmeans_df.iloc[:,1:6]

The next command plots our clusters. It uses a technique called Principal Component Analysis (https://en.wikipedia.org/wiki/Principal_component_analysis), which we can ignore now, but will revisit later.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# this doesn't look the same, where are all the dots, which words are used in the original assignment?
# it also doesn't look as if that is all of them

pca = PCA(2)
data = pd.DataFrame(pca.fit_transform(kmeans_df), columns = ['PC1', 'PC2'])

data['cluster'] = pd.Categorical(kmeans.labels_)
sns.scatterplot(x="PC1",y="PC2",hue="cluster",data=data)

data

