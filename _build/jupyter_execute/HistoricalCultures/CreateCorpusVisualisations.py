# Create corpus and Visualise

## Create Corpus

In the this exercise you will build your own corpus. It is easy.

Please, create a folder on your computer to download 5-10 of the top Gutenberg books from https://www.gutenberg.org/browse/scores/top 

You need to download the Plain-Text versions (ASCII).

Set my_folder to the directory you used to download the books. Do you remember the relative and absolute paths and how to set your working directory?

# mycorpus laden

Wonderland = open("MyCorpus/11-0.txt", "r")
Pride = open("MyCorpus/1342-0.txt", "r")
Tale = open("MyCorpus/98-0.txt", "r")
Yellow = open("MyCorpus/pg1952.txt", "r")

Wonderland = Wonderland.read()
Pride = Pride.read()
Tale = Tale.read()
Yellow = Yellow.read()

Now we simply use NLTK to clean the text and to create a corpus out of these texts.

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

Corpuser(Wonderland)
Corpuser(Pride)
Corpuser(Tale)
Corpuser(Yellow)

Success! Let's look at the content of the second book. 

print(Pride)

### Create DocumentTermMatrix

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


ft_won = frequencytable(Wonderland)
ft_prid = frequencytable(Pride)
ft_tale = frequencytable(Tale)
ft_yell = frequencytable(Yellow)



# create dataframe from dict
import pandas as pd

df_won = pd.DataFrame.from_dict(ft_won, orient='index', columns={'wonderland'})
df_prid = pd.DataFrame.from_dict(ft_prid, orient='index', columns={'Pride'})
df_tale = pd.DataFrame.from_dict(ft_tale, orient='index', columns={'Tale'})
df_yell = pd.DataFrame.from_dict(ft_yell, orient='index', columns={'Yellow'})

merged_df = pd.concat([df_won, df_prid, df_tale, df_yell], axis=1)
merged_df = merged_df.fillna(0)

merged_df

You can now do any of the advanced processing we discussed. Here, we will simply add a few visualisations, starting with word clouds.

## Visualise Texts


### Wordclouds


from wordcloud import WordCloud
import matplotlib.pyplot as plt

corpus = str(Corpuser(Wonderland) + Corpuser(Pride) + Corpuser(Tale) + Corpuser(Yellow))
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

#  comparison cloud not available in python, we could plot two wordclouds from different texts

corpus = str(Corpuser(Wonderland))
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                min_font_size = 10,
                colormap='RdYlGn').generate(corpus) 

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

corpus = str(Corpuser(Tale))
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                min_font_size = 10).generate(corpus) 

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 



## Plot term frequencies

# Wonderland

sort = df_won.sort_values(by=['wonderland'], ascending=False)

top15 = sort[:15]

top15.plot.bar()

# Word networks?