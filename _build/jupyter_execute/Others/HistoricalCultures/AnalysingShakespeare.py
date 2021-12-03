# Analysing Shakespeare

In this exercise, we will analyse Shakepeare using our new knowledge on text miming.

First we need to download the complete collection of all Skapespeare's works. We can easily do this using NLTK.

import nltk
nltk.corpus.gutenberg.fileids()

Finally, let's create a Shakespeare corpus and investigate the texts

#  other text but does that matter?

ceasar = nltk.corpus.gutenberg.words('shakespeare-caesar.txt')
hamlet = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
macbeth = nltk.corpus.gutenberg.words('shakespeare-macbeth.txt')

Next, we apply our usual text transformations, starting with tokenizing the words, stripping the whitespaces, lower-casing the words, removing numbers and removing stopwords

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopwords = list(stopwords.words('english'))

def Corpuser(corpus):
#     corpus = word_tokenize(corpus)
    corpus = [word.replace(" ", "") for word in corpus]
    corpus = [word.lower() for word in corpus if word.isalpha()]

    corpus = [word for word in corpus if word not in stopwords]
    
    return corpus

ceasar_corp = Corpuser(ceasar)
hamlet_corp = Corpuser(hamlet)
macbeth_corp = Corpuser(macbeth)

We then create a DocumentTermMatrix

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 

docs = [str(ceasar_corp), str(hamlet_corp), str(macbeth_corp)]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)


dtm = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
dtm

There are many many empty entries (0), which means the words do not appear in the document. We can remove terms that appear less then a certain amount with min_df as a parameter in CountVectorizer

vectorizer = CountVectorizer(min_df = 2)
X = vectorizer.fit_transform(docs)
dtm = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
dtm

Better, much better ...

Let's find the most frequent terms in the three Shakespeare docs next.

freqdist = FreqDist(ceasar_corp+hamlet_corp+macbeth_corp)
freqdist.most_common(10)

## TF/IDF

We have discussed the tf/idf scoring for documents in the lecture. 

In python, tf/idf is easy to create using scikit learn. 

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=2)
vec = tfidf.fit_transform(docs)
 
matrix = pd.DataFrame(vec.toarray().transpose(), index=tfidf.get_feature_names())
 
matrix[:10].T

## k-means clustering

Using, tf-idf we can apply our favourite clustering technique k-means to understand common clusters of words.

We a technique called Principal Component Analysis (https://en.wikipedia.org/wiki/Principal_component_analysis), which we can ignore now, but will revisit later.

from sklearn.decomposition import PCA

pca = PCA(2)
 
df = pca.fit_transform(matrix)
 
df.shape

Which word belongs to which cluster?

from sklearn.cluster import KMeans
 
kmeans = KMeans(n_clusters= 3)
 
label = kmeans.fit_predict(df)
 
matrix['cluster'] = label
matrix

The next command plots our clusters

import seaborn as sns
sns.scatterplot(x=df[:,0],y=df[:,1], hue=label)

### Topic Models

Let's try topic models next, because we know how and they are easy to do ...

import gensim
import gensim.corpora as corpora

corpor = [ceasar_corp, hamlet_corp, macbeth_corp]
Dict = corpora.Dictionary(corpor)
texts = corpor

# Term Document Frequency
td = [Dict.doc2bow(text) for text in texts]

print(td)

We target 10 topics.

n = 10

LDA = gensim.models.LdaMulticore(corpus=td,
                                       id2word=Dict,
                                       num_topics=n)

What do the terms per topic look like?

The weights reflect how important a keyword is to that topic.

from pprint import pprint

pprint(LDA.print_topics())