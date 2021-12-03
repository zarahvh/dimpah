# R graphs and Gephi

This exercise will teach you how to use Python together with the popular visualisation environment Igraph. 
The paper you read today, used both to perfection: Pablo Barbera, John T. Jost, Jonathan Nagler, Joshua Tucker, and Richard Bonneau: "Tweeting from Left to Right: Is Online Political Communication More Than an Echo Chamber?".

Check out the paper's code at https://github.com/pablobarbera/echo_chambers and in particular https://github.com/pablobarbera/echo_chambers/blob/master/03_analysis/14-network-visualization.r, which produces all the wonderful network visualisations you have seen in the text. If we would have access to the data we could just rerun the analysis.

The exercise today will teach you how to produce such Igraph graphs based on an Python analysis. Well, we will take the first steps at least.

## Delitsch School Class Network

Our first example is the friendship network of a German boys' school class from 1880/1881. It's based on the probably first ever collected social network dataset, assembled by the primary school teacher Johannes Delitsch. The data was reanalyzed and compiled for the article: Heidler, R., Gamper, M., Herz, A., Eßer, F. (2014): Relationship patterns in the 19th century: The friendship network in a German boys' school class from 1880 to 1881 revisited. Social Networks 13: 1--13..

Let's load igraph first.

import igraph
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import json

## The Data
The Delitsch data is prepared for you. You can load it with igraph.read() from igraph.

g = igraph.read("Delitsch-Network.gml ", format="gml")

Check out the vertices (nodes) labels and create a quick plot of the graph.

from igraph import *

summary(g)
print(g)

import cairocffi
# layout = g.layout("kk")
plot(VertexClustering(g))

### Simple Statistics with Python igraph

To demonstrate that you can use Python as a statistical engine to support your Graphi work, we now run a few simple statistics.

# node degree
node_degree = g.degree(mode='all')
# degree distribution
degree_distr = g.degree_distribution(mode='all')

# ??
plt.plot(node_degree)

### Change the graph

Next we update the graph to prepare it for its Gephi visualisation. You should know already how this works.

plot(g, layout = layout, vertex_color='orange', edge_arrow_size=.2, edge_color='grey', edge_curved=False)

### Hubs/Authorities

Let's do one more example analysis. Network hubs are expected to contain vertices with a large number of outgoing links, while authorities would get many incoming links from hubs.

Let's plot both.

hs = g.hub_score()
asc = g.authority_score()

plot(g, layout = layout, vertex_color='orange', vertex_size=g.hub_score()*20, edge_color='grey', edge_curved=False)

plot(g, layout = layout, vertex_color='orange', vertex_size=g.authority_score()*20, edge_color='grey', edge_curved=False)

## Twitter Network Analysis

The Delitsch network is quite interesting, but maybe you are more excited by being able to visualise Twitter networks?

First we do our Twitter thing and set up the environment for searching Twitter from R. I have defined a token for you and use this opportunity to show you how to load it.

import requests
from ipynb.fs.full.keys import *
import tweepy

consumer_key = twit_key
consumer_secret = twit_secr
access_token = twit_token

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

Now we search Twitter for the 'truth'. Up to 100 recent Tweets.

import tweepy as tw
tweets = tw.Cursor(api.search, 
                           q='truth',
                           lang="en").items(100)

users_text = [[tweet.user.screen_name, tweet.text, tweet.user.followers_count, tweet.user.favourites_count] for tweet in tweets]
# users_text

tweet_df = pd.DataFrame(data=users_text, 
                    columns=['user', "text", 'followers', 'favourites_count'])
tweet_df

# edges <- gt_edges(tweets, text, screen_name, status_id)
# #edges <- edges[!duplicated(edges),]
# nodes <- gt_nodes(edges, meta = TRUE)

Let's plot a quick histogram of followers_count to demonstrate what we could do with it.

tweet_df['followers'].plot.hist()

# Collect edges and nodes with the count for the favourite
# but what are the relations? the nodes are the users but what is the edge, is it the amount of followers?
nodes = pd.DataFrame(columns=['user_name'])
# nodes['userid'] = graphdata.follower.unique()
nodes['user_name'] = tweet_df.user.unique()

relations = pd.DataFrame(columns=['followers'])
relations['followers'] = tweet_df['followers']
# relations['follower'] = graphdata['follower']



graph = Graph.DictList(
          vertices=nodes.to_dict('records'),
          edges=relations.to_dict('records'),
          directed=True,
          vertex_name_attr='user_name',
          edge_foreign_keys=('followers'));





