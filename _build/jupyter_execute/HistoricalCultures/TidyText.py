In 2016, David Robinson's published a great analysis of Donald Trump's (http://varianceexplained.org/r/trump-tweets/). It got a lot of publicity and his collaboration with Julia Slige resulted in a new book and approach called tidytext (http://tidytextmining.com/). 

https://github.com/juliasilge/tidytext has become another package for advanced text analysis that has quickly gained a lot of support.

The tidytext package allows to use tidytext principles (https://www.jstatsoft.org/article/view/v059i10) with unstructured data/text.

Let's take a character vector with one element made of 3 sentences.

import pandas as pd

text = """
Using tidy data principles is important.
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

And tokenize into phrases (bigrams).

# bigrams not available in pyhton function, is there a way around it?

## Removing stopwords

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopwords = list(stopwords.words('english'))

new_table = table[~table['word'].isin(stopwords)]
new_table

## Summarizing word frequencies

# Count function using nltk too? is it available in tidytext?
# bind_tfidf what does it do?

def frequencytable(df):
    words = df['word']
    freq_table = {}
    for word in words:
        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1
    return freq_table

frequencytable(new_table)

## Case Study Gutenberg

### Gutenbergr

The gutenberg package (https://ropensci.org/tutorials/gutenbergr_tutorial.html) provides access to the Project Gutenberg collection. The package contains tools for downloading books and for finding works of interest.

import gutenberg

# sherlock holmes

# Retrieve the first 10 titles of Arthur Conan Doyle in the Gutenberg library.

# how to get the books? either download them beforehand or use beautifulsoup?

# removing stopwords and word count can be done using the previous functions again
# ggplot in python
# sentiment analysis --> again using NLTK?

