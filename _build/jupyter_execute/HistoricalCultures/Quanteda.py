# something about nltk's corpora, what they do, what they can be used for etc.

from nltk.corpus import brown

print(brown.fileids())

We can read these files as either raw text, a list of words, a list of sentences, or a list of paragraphs.

brown.raw('ca01')

brown.words('ca01')

brown.sents('ca01')

brown.paras('ca01')

Another special feauture that the brown corpus has is that it's sentences are already tagged

brown.tagged_sents('ca01')

The same goes for the words

brown.tagged_words('ca01')

We can also look in the brown corpus for certain categories instead of files such as the news

brown.words(categories='news')

## Document Term Matrix

We can again construct a document term matrix, let's do this for the first three files in the brown corpus

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 

doc1 = brown.words('ca01')
doc2 = brown.words('ca02')
doc3 = brown.words('ca03')

docs = [doc1, doc2, doc3]
docnew = []
for doc in docs:
    doc_str = " ".join(doc)
    docnew.append(doc_str)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docnew)

dtm = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
dtm

As you can see the corpus still contains numbers and uppercase letters, so let's use some of the functions we used before to clean the corpus

doc1  = [word.lower() for word in doc1 if word.isalpha()]
doc2  = [word.lower() for word in doc2 if word.isalpha()]
doc3  = [word.lower() for word in doc3 if word.isalpha()]

doc_clean = []
for doc in docs:
    doc = [word for word in doc if word.isalpha()]
    doc_str = " ".join(doc)
    doc_clean.append(doc_str)
    
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(doc_clean)

dtm = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
dtm

That's better

## Wordcloud

As we have done before we can also construct a wordcloud

import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                min_font_size = 10).generate(str(doc_clean)) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

