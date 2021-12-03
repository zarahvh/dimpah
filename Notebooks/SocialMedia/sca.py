import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud


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

# Social Sensing 1


def make_wordcloud_tweets(tweet_text):
    strings = ''.join(tweet_text)
    corpus = word_tokenize(strings)
    corpus = [word.replace(" ", "") for word in corpus]
    corpus = [word.lower() for word in corpus if word.isalpha()]
    stopwords_ = list(stopwords.words('english'))
    corpus = [word for word in corpus if word not in stopwords_]
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                min_font_size = 10).generate(str(corpus)) 
  
    # plot the WordCloud image
    f, ax = plt.subplots(figsize = (8, 8), facecolor = None)
    ax.imshow(wordcloud) 
    plt.axis("off") 
    
    return ax


def create_twitter_network(sn_list):
    G = nx.Graph()
    twitter_network_df = pd.DataFrame(sn_list, columns=['source', 'target'])
    G = nx.from_pandas_edgelist(twitter_network_df)
    d = dict(G.degree)
    
    f, ax = plt.subplots(figsize = (20, 20))
    pos = nx.spring_layout(G)
    
    nx.draw(G, pos, edge_color='lightgrey', node_color = 'tomato',
            node_size=[v * 100 for v in d.values()], with_labels=True)
    return ax



# Social Sensing 2  

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
