import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import tweepy

from matplotlib.figure import Figure
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

#https://stackoverflow.com/questions/43541376/how-to-draw-communities-with-networkx
def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

def plot_twitter_activity(tweets):
    fig = Figure()
    dates = [tweet.created_at.date() for tweet in tweets]
    counts = Counter(dates)
    plt.bar(counts.keys(), counts.values())
    return fig


def plot_wordcloud(words):
    strings = ''.join(words)
    corpus = word_tokenize(strings)
    corpus = [word.replace(" ", "") for word in corpus]
    corpus = [word.lower() for word in corpus if word.isalpha()]

    stop_words = list(stopwords.words('english'))
    corpus = [word for word in corpus if word not in stop_words]
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white',  
                    min_font_size = 10).generate(str(corpus)) 

    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.show() 


# +
# Get list of al the follower ids (https://towardsdatascience.com/how-to-download-and-visualize-your-twitter-network-f009dbbf107b)
    
def get_follower_list(user, twit_key):
    if twit_key != '':
        user_id = user.id
        user = [user_id]
        follower_list = []
        for user in user:
            followers = []
            try:
                for page in tweepy.Cursor(api.followers_ids, user_id=user).pages():
                    followers.extend(page)
            except tweepy.TweepError:
                print("error")
                continue
            follower_list.append(followers)
    else:
        df = pd.read_csv('followers_tobias.csv')
        follower_list = df['followers'].to_list()
    return follower_list


# -

def get_follower_count(followers,n):
    fol_count = []
    for follower in followers[:n]:
        try:
            user = api.get_user(follower)
            count = user.followers_count
            fol_count.append(count)
        except tweepy.TweepError:
            fol_count.append(0)
            print("error")
            continue
    return fol_count


def get_friends_count(user, followers):

    friends_count = []
    for follower in followers:
        try:
            user = api.get_user(follower)
            count = user.friends_count
            friends_count.append(count)
        except tweepy.TweepError:
            friends_count.append(0)
            print("error")
            continue
    
    return friends_count


# +
def get_friends_of20(user):
    n = 1
    for user in user:
        friends = []
        screen_name_friends = []
        try:
            if n != 20:
                for friend in tweepy.Cursor(api.friends_ids, user_id=user).items():
                    n+=1
                    friends.append(friend)
                    user = api.get_user(friend)
                    name = user.screen_name
                    screen_name_friends.append(name)
        except tweepy.TweepError:
            friends.append('error')
            print("error")
            continue


