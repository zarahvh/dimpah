import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import requests
import json
import os

import geopandas as gpd 
import geopy 
import matplotlib.pyplot as plt
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium.plugins import FastMarkerCluster

import spacy

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/opinion_lexicon')
except LookupError:
    nltk.download('opinion_lexicon')
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')   

if (spacy.util.is_package("en_core_web_sm") == False):
     os.system('spacy download en_core_web_sm') 

#Historical Cultures 2



def google_ngram(term, start_year, end_year):
    url_str = 'https://books.google.com/ngrams/json?content=' + term + '&year_start=' + str(start_year) + '&year_end=' + str(end_year) + '&corpus=26&smoothing=3'
    result = requests.get(url = url_str)
    json_data = json.loads(result.text)
    data = json_data[0]
    df = pd.DataFrame.from_dict(data['timeseries'])
    df = df.rename(columns={0: term})
    df['year'] = np.arange(start_year, end_year+1, 1) 
    return df



def locations_spacy(doc):
    locator = geopy.geocoders.Nominatim(user_agent='mygeocoder')
    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
    locations = [ent for ent in doc.ents if ent.label_ in ['GPE']]
    locations_geocoded = []
    for l in locations:
        try:
            locations_geocoded.append(geocode(l).point)
        except:
            continue
    locations_geocoded = [(l[0], l[1]) for l in locations_geocoded]
    fm = folium.Map(location=[38.305542, -30.384108], tiles='cartodbpositron', zoom_start = 2) 
    FastMarkerCluster(data=locations_geocoded).add_to(fm)
    return fm
