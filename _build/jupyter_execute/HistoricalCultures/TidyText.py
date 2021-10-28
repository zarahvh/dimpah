# TidyText

In 2016, David Robinson's published a great analysis of Donald Trump's (http://varianceexplained.org/r/trump-tweets/). It got a lot of publicity and his collaboration with Julia Slige resulted in a new book and approach called tidytext (http://tidytextmining.com/). 

https://github.com/juliasilge/tidytext has become another package for advanced text analysis that has quickly gained a lot of support.

The tidytext package allows to use tidytext principles (https://www.jstatsoft.org/article/view/v059i10) with unstructured data/text.

Let's take a character vector with one element made of 3 sentences.

import pandas as pd

text = """Using tidy data principles is important.
In this package, we provide functions for tidy formats.
The novels of Jane Austen can be so tidy!
"""

The dataset is not yet compatible with the tidy tools. The first step is to use unnest.

### unnest_tokens function

The unnest_token function splits a text column (input) into tokens (e.g. sentences, words, ngrams, etc.).

text_split = text.splitlines()

df = pd.DataFrame({
    "text": text_split,
    "line": list(range(len(text_split)))
})

Next for the tidy text format.

### The tidy text format

Tidy text format is define as 'a table with one-term-per-row'. 

To tokenize into words (unigrams).

from tidytext import unnest_tokens

table = unnest_tokens(df, "word", "text")
table

## Removing stopwords

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopwords = list(stopwords.words('english'))

new_table = table[~table['word'].isin(stopwords)]
new_table

## Summarizing word frequencies

from nltk.probability import FreqDist

words = new_table['word'].values
FreqDist(words).most_common(20)

## Case Study Austen

### Gutenbergr

We will use the Gutenberg package from NLTK again to this time analyse the texts of Jane Austen


import nltk
nltk.corpus.gutenberg.fileids()

emma = nltk.corpus.gutenberg.sents('austen-emma.txt')
persuasion = nltk.corpus.gutenberg.words('austen-persuasion.txt')
sense = nltk.corpus.gutenberg.words('austen-sense.txt')

Transform into a tidy dataset..

sense

def to_tidy(corp):
    new = []
    for word in corp:
        string = " ".join(word)
        new.append(string)

    corp_str = ''
    for sent in new: 
        corp_str += sent

    text_split = corp_str.split('.')

    df = pd.DataFrame({
        "text": text_split,
        "line": list(range(len(text_split)))
    })
    return df

emma_df = to_tidy(emma)
emma_table = unnest_tokens(emma_df, "word", "text")

emma_table

Remove stopwords

emma_table = emma_table[~emma_table['word'].isin(stopwords)]

emma_table

Calculate frequencies

words = new_table2['word'].values
FreqDist(words).most_common(20)

## Sentiment analysis Austen

We can perform a sentiment analysis on these texts with NLTK.

NLTK has a built-in sentiment analyzer: VADER (Valence Aware Dictionary and sEntiment Reasoner).

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sia.polarity_scores(str(emma_new))

We can check for each sentence whether it is positive or negative

# from:  https://www.codeproject.com/Articles/5269445/Using-Pre-trained-VADER-Models-for-NLTK-Sentiment
scores = {'pos': 0, 'neg': 0, 'neu': 0}
for sent in emma_new:
    score = sia.polarity_scores(sent)
    if score['pos'] > 0.5:
        result['pos'] += 1
    elif score['neg'] > 0.5:
        result['neg'] += 1
    elif score['neu'] > 0.5:
        result['neu'] += 1
print(scores)

As we can see most sentences in Emma are neutral