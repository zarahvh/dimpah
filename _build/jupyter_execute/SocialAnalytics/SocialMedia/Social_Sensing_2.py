#  Social Sensing 2

We have just experienced the power of social networks using python. You have seen how powerful the libraries are and have worked through a complete Twitter analysis. But we have also seen how difficult it is to get the graph visualisations right. The plotted networks can definitely improve. Other plots are fairly boring – like my uninspiring Twitter follower network. In order to dig deeper into the details of social network analysis, let’s use a famous example from the past.

Zachary’s karate club is a social network of friendships of 34 members of a karate club at a US university in the 1970s. It is described in W. W. Zachary, An information flow model for conflict and fission in small groups, Journal of Anthropological Research 33, 452-473 (1977). It is a famous early example of successful social network analysis. According to Wikipedia, ‘during the study a conflict arose between the administrator and instructor, which led to the split of the club into two. Half of the members formed a new club around the instructor members from the other part found a new instructor or gave up karate. Based on collected data Zachary assigned correctly all but one member of the club to the groups they actually joined after the split.’ https://en.wikipedia.org/wiki/Zachary%27s_karate_club.

--> igraph python https://igraph.org/python/

First download igraph.

You have two data frames loaded in the environment. karate_nodes contains the nodes of the network with information about the karate club members. Check it with head(karate_nodes).

import igraph
import pandas as pd
import numpy as np
import csv
import matplotlib as plt

karate_nodes = pd.read_csv("data/karate_nodes.csv")
karate_nodes.head()

The second data frame karate_edges contains the edges with information whether one member likes another and by how much.

karate_edges = pd.read_csv("data/karate_edges.csv")
karate_edges[:20]

Now, let’s create an igraph karate_g with these edges and nodes.

We first need to download iGraph and then type from igraph import*

A tutorial can be found at: https://igraph.org/python/doc/tutorial/tutorial.html#layouts-and-plotting

from igraph import *
g = Graph()

# Igraph needs tuples, cannot work with a dataframe directly

tuple_edges = [tuple(x) for x in karate_edges.values]

Gm = igraph.Graph.TupleList(tuple_edges, directed = True, edge_attrs = ['weight'])

tuple_nodes = [tuple(x) for x in karate_nodes.values]
Gm.add_vertices(tuple_nodes)

print(Gm)

graph = Graph.DictList(
          vertices=karate_nodes.to_dict('records'),
          edges=karate_edges.to_dict('records'),
          directed=True,
          vertex_name_attr='id',
          edge_foreign_keys=('from', 'to'));

print(graph)

We can look into the nodes of the graph using graph.vs

vseq = graph.vs
for v in vseq:
    print(v)

We can see the node together with its attributes, if we want to now only see the age attribute of each node, we access it using ['age']

for v in vseq:
    print(v['age'])

Let's try an plot this graph

(We need to install cairo to do so)

pip3 install cairocffi

import cairocffi
layout = graph.layout("kamada_kawai")
plot(graph, layout = layout)

You can also plot a graph with curved edges (edge_curved=True) and reduced arrow size. (edge_arrow_size=.4)

layout = graph.layout("kamada_kawai")
plot(graph, layout = layout, edge_curved=True, edge_arrow_size=.4)

Maybe that is a bit much let's change the curve to 0.1. Let’s try a more complicated plot, we could also add some other parameters. Now we can access the ids by using graph.vs['id'] and add them as vertex_label parameter.

layout = graph.layout("kamada_kawai")
plot(graph, layout = layout, edge_curved=True, edge_arrow_size=.4, vertex_label=graph.vs['id'], vertex_color='orange')

We could do the same, but then with the actual names, remember how we accessed the ids? Do the same with the names.

layout = graph.layout("kamada_kawai")
plot(graph, layout = layout, edge_curved=True, edge_arrow_size=.4, vertex_label=graph.vs['first_name'], vertex_color='orange')

 Let’s say we want to colour our network nodes based on gender as well as size them based on age. We will also change the width of the edges based on their weight. We need to apply a series of changes to the attributes pf our igraph. To get the different colors for different genders we need to add these colors to our dataframe karate_nodes.

karate_nodes['color'] = np.where(karate_nodes['gender']== 'F', 'red', 'blue')
karate_nodes['age'] = pd.to_numeric(karate_nodes['age'])

karate_nodes.head()

And then recreate the graph and check the vseq again whether color is now also an attribute

graph = Graph.DictList(
          vertices=karate_nodes.to_dict('records'),
          edges=karate_edges.to_dict('records'),
          directed=True,
          vertex_name_attr='id',
          edge_foreign_keys=('from', 'to'));

We then plot the graph using the graph.vs of the colors the same as we did with the ids

layout = graph.layout("kamada_kawai")
plot(graph, layout = layout, edge_curved=True, edge_arrow_size=.4, vertex_label=graph.vs['first_name'], vertex_color=graph.vs['color'])

We also wanted to change the node size based on age. We could, for example, multiply the age by 0.4 to get the size that we want. We can do this more easily by setting vertex_size= [v*0.4 for v in graph.vs['age']].

*Could we have done the same for color?

plot(graph, layout = layout, edge_curved=True, edge_arrow_size=.4, vertex_label=graph.vs['first_name'], vertex_color=graph.vs['color'], vertex_size= [v*0.4 for v in graph.vs['age']])


The weight of the like-relationship will determine the width of the arrow between two nodes. But check, this is an edge attribute instead of a vertix attribute, we can access this by using graph.es['weight'], now let's do the same as above but then for the edge.

plot(graph, layout = layout, edge_curved=True, edge_arrow_size=.4, vertex_label=graph.vs['first_name'], vertex_color=graph.vs['color'], vertex_size= [v*0.4 for v in graph.vs['age']], edge_width = [e/5 for e in graph.es['weight']])

You can also plot graphs with different layouts.
To adjust the graph layout, igraph contains layout generators, which try to place the vertices and edges in a way that is more visually appealing. There are many layout functions, let's first try a random one.

layout = graph.layout("random")
plot(graph, layout = layout, edge_curved=True, edge_arrow_size=.4, vertex_label=graph.vs['first_name'], vertex_color=graph.vs['color'], vertex_size= [v*0.4 for v in graph.vs['age']], edge_width = [e/5 for e in graph.es['weight']])

Maybe this one is a bit too curved in the edges which makes it unreadable, let's get rid of that argument

(Not sure, maybe the curved edges are better, what do you think?)

plot(graph, layout = layout, edge_arrow_size=.4, vertex_label=graph.vs['first_name'], vertex_color=graph.vs['color'], vertex_size= [v*0.4 for v in graph.vs['age']], edge_width = [e/5 for e in graph.es['weight']])

We can now change the layout parameter and use another function. Fruchterman Reingold (https://en.wikipedia.org/wiki/Force-directed_graph_drawing) is a very popular layout algorithm. Type in graph.layout("fr")

layout = graph.layout("fr")
plot(graph, layout = layout, edge_arrow_size=.4, vertex_label=graph.vs['first_name'], vertex_color=graph.vs['color'], vertex_size= [v*0.4 for v in graph.vs['age']], edge_width = [e/5 for e in graph.es['weight']])


This is much much better. But remember the original insight from the 1970s paper of the karate club? It described how the larger community of the whole club was effectively the result of several separate communities of members and split thererfore according to trust placed in either the administrator or instructor. Thus the whole karate club community can split up easily. Graph analysis comes with a lot of so-called community detection algorithms that support such investigations

(Possible tutorial: https://towardsdatascience.com/detecting-communities-in-a-language-co-occurrence-network-f6d9dfc70bab?)

com = graph.community_walktrap()
clp = com.as_clustering()

Now let's see what's behind clp

print(clp)

The result is a list with similar information to the one we have already met during the clustering exercises. We can easily plot the communities with plot(clp, mark_groups=True).

plot(clp, mark_groups=True, vertex_label=graph.vs['first_name'])

This graph already indicates that some members hold the whole network together by being the main link between the various 4 communities walktrap has detected. Let’s investigate this further and visualise the degree by which members are connected to other members. A graph degree basically counts the number of connections a member has to other members. Let’s overwrite the size of each nodes with the degree. First though we need to calculate the degree for each node. That’s very easy using igraph’s degree function. Simply type graph.degree()

graph.degree()

Now, let’s reassign the size of the nodes with setting vertex_size= vertex_size=graph.degree()

plot(clp, mark_groups=True, vertex_label=graph.vs['first_name'], vertex_size=graph.degree())

The new graph clearly shows now where the potential breaking points in the network are. Social network analysis is a very powerful tool with a large community already out there. Check it out and happy playing!

Let’s check what we have learned now.

What is a social network?

1. A movie
2. All my friends
3. Something other people know more about than me
4. A social structure made up of a set of social actors and their interactions

A social structure made up of a set of social actors and their interactions

Double the edge width of the graph (without the communities) based on the value we have set earleier based on weight.

plot(graph, vertex_label=graph.vs['first_name'], edge_width = [e/2.5 for e in graph.es['weight']])

Change the size of the nodes


plot(graph, vertex_label=graph.vs['first_name'], edge_width = [e/2.5 for e in graph.es['weight']],vertex_size= 50)

Run another community detection algorithm graph.community_label_propagation()

prop= graph.community_label_propagation()

plot(prop, mark_groups=True)

That’s it for today. You have learned a lot of things about how to create social sensing networks. This is one of the most important social analytics techniques, and you can impress friends and family now with pretty graphs using the structure of social networks. Next time, we will look into content analysis using text mining.