# Create corpus and Visualise

## Create Corpus

In the this exercise you will build your own corpus. It is easy.

Please, create a folder on your computer to download 5-10 of the top Gutenberg books from https://www.gutenberg.org/browse/scores/top 

You need to download the Plain-Text versions (ASCII).

Set my_folder to the directory you used to download the books. Do you remember the relative and absolute paths and how to set your working directory?

Wonderland = open("MyCorpus/11-0.txt", "r")
Pride = open("MyCorpus/1342-0.txt", "r")
Tale = open("MyCorpus/98-0.txt", "r")
Yellow = open("MyCorpus/pg1952.txt", "r")

Wonderland = Wonderland.read()
Pride = Pride.read()
Tale = Tale.read()
Yellow = Yellow.read()

### Clean corpus

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

won_corp = Corpuser(Wonderland)
pride_corp = Corpuser(Pride)
tale_corp = Corpuser(Tale)
yel_corp = Corpuser(Yellow)

Success! Let's look at the content of the second book. 

print(Pride)

### Create DocumentTermMatrix

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 

docs = [str(won_corp), str(pride_corp), str(tale_corp), str(yel_corp)]
vectorizer = CountVectorizer(min_df=2)
X = vectorizer.fit_transform(docs)

dtm = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
dtm

You can now do any of the advanced processing we discussed. Here, we will simply add a few visualisations, starting with word clouds.

## Visualise Texts


### Wordclouds

from wordcloud import WordCloud
import matplotlib.pyplot as plt

corpus = str(won_corp + pride_corp + tale_corp + yel_corp)
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                min_font_size = 10).generate(corpus) 

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()

The commonality wordcloud visualises common words across documents, which is of couse in this small corpus identical to the normal word cloud.

Try to change the colours!

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                min_font_size = 10,
                colormap='RdYlGn').generate(corpus) 

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

We can also compare the wordclouds of two books

corpus = str(won_corp)
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                min_font_size = 10,
                colormap='RdYlGn').generate(corpus) 

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

corpus = str(tale_corp)
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                min_font_size = 10).generate(corpus) 

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

### Most frequent terms

from nltk import *

freqdist = FreqDist(won_corp+pride_corp+tale_corp+yel_corp)
freqdist.most_common(15)

freqdist.plot(15)