# Eighteenth Century beta of PRISM 

This is a simple but effective example of the power of social network analysis using the principles of using connection to detect 'suspicous activity' in the network of American revolutionaries in the 18th century before the American revolution. We will play with the principles of the NSA's PRISM programme https://en.wikipedia.org/wiki/PRISM_(surveillance_program) but apply these to the 18th century.

The code is adapted from https://kieranhealy.org/blog/archives/2013/06/09/using-metadata-to-find-paul-revere/ and
https://github.com/kjhealy/revere/blob/master/revere.R. Check out Kieran's wonderful work: https://kieranhealy.org/

Kieran introduces his ideas with: 'A brief demonstration of the surprising effectiveness of even the simplest techniques of the new-fangled Social Networke Analysis in the pursuit of those who would seek to undermine the liberty enjoyed by His Majesty’s subjects. This is in connection with the discussion of the role of “metadata” in certain recent events and the assurances of various respectable parties that the government was merely “sifting through this so-called metadata” and that the “information acquired does not include the content of any communications”. I will show how we can use this “metadata” to find key persons involved in terrorist groups operating within the Colonies at the present time (...).' (https://kieranhealy.org/blog/archives/2013/06/09/using-metadata-to-find-paul-revere/).

Let's load the library igraph next, which you should already have installed. Otherwise, please install it first.

import igraph as ig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

The data for the analysis is based on the appendix of David Hackett Fischer: Paul Revere's ride. Oxford University Press, 1995. Kieran has done the work of creating the dataset for us. Let's load the data from his github pages.

Paul = pd.read_csv('PaulRevereAppD.csv', index_col=0)
Paul.head()

Next we create an adjacency matrix (https://en.wikipedia.org/wiki/Adjacency_matrix) using matrix multiplication (%*%). This is quite basic matrix stuff, but you do not need to know how this is exactly calculated. If you are nevertheless interested, please check out https://en.wikipedia.org/wiki/Matrix_multiplication. t is the transpase operator in R (https://en.wikipedia.org/wiki/Transpose). We will create two new adjacency matrixes. The first one counts the number of times a person is in the same group with another person. The second matrix counts the number of members per group. 

You could easily implement the count of groups and people with, for instance, a for loop but matrix multiplication is much more efficient. 

#Count the number of groups two persons are part of!
person = Paul.dot(Paul.T)
person.head()

#Members in a group
group = Paul.T.dot(Paul)
group.head()

Next we do some clean up and define all the matrix diagonals as NA, as they are meaningless and transform the dataframe into one that we can use for the graph

np.fill_diagonal(group.values, 333)
group = group.replace(333, np.nan)

groun = group.stack().reset_index()
groun.columns = ['from','to','weight']
groun.head()

nodes = pd.DataFrame(columns=['group'])
nodes['group'] = groun['from'].unique()
nodes['weight'] = groun['weight']

relations = pd.DataFrame(columns=['source'])
relations['source'] = groun['from']
relations['target'] = groun['to']

G_group = ig.Graph.DictList(
          vertices=nodes.to_dict('records'),
          edges=relations.to_dict('records'),
          directed=True,
          vertex_name_attr='group',
          edge_foreign_keys=('source', 'target'));

import cairocffi
layout = G_group.layout("Fruchterman_reingold")
ig.plot(G_group, layout = layout, vertex_label=G_group.vs['group'], edge_curved=False, edge_width = [v for v in G_group.vs['weight']])

np.fill_diagonal(person.values, 333)
person = person.replace(333, np.nan)

personn = person.stack().reset_index()
personn.columns = ['from','to','weight']
person_df = personn[personn.weight != 0]
person_df.head()

nodes = pd.DataFrame(columns=['person'])
nodes['person'] = person_df['from'].unique()
nodes['weight'] = person_df['weight']
nodes['color'] = 'blue'
nodes.loc[nodes['person'] == 'Revere.Paul', 'color'] = 'red'

relations = pd.DataFrame(columns=['source'])
relations['source'] = person_df['from']
relations['target'] = person_df['to']
relations['weight'] = person_df['weight']

G_person = ig.Graph.DictList(
          vertices=nodes.to_dict('records'),
          edges=relations.to_dict('records'),
          directed=False,
          vertex_name_attr='person',
          edge_foreign_keys=('source', 'target'));

Plotting the people in the data and their connections, reveals ...? We mark Paul Revere in red.

import cairocffi
layout = G_person.layout("fr")
ig.plot(G_person, layout=layout, vertex_label=G_person.vs['person'], edge_curved=False, edge_width = [v*0.08 for v in G_person.vs['weight']], edge_arrow_size=0.2, vertex_size=15)

## Finding Paul Revere

Next we apply a number of typical graph measures to get clarity about a complex networks such as the one above. Don't worry if you do not understand them immediately. You can find many explanations on the web that should help you. If after this you have an intuition that should be enough.

### Betweeness

Betweenness is a measure of the centrality of everyone in our graph, which is roughly the number of "shortest paths" between any two people in our network that pass through the person of interest. Or put more simply: "If I have to get from person a to person z, how likely is it that the quickest way is through person x?"

betw = G_person.betweenness()
nodes['betweenness'] = betw
betw_sorted = nodes.sort_values(by=['betweenness'], ascending=False)
betw_sorted[:10]

### Eigenvector

The Eigenvector on the other hand measures centrality weighted by a person's connection to other central people.

eg = G_person.eigenvector_centrality()
nodes['eg'] = eg
eg_sorted = nodes.sort_values(by=['eg'], ascending=False)
eg_sorted[:10]

### Community Detection

Finally, let's try community detection to again find Paul Revere at the centre.

com = G_person.community_walktrap()
clp = com.as_clustering()

ig.plot(clp, layout=layout, mark_groups=True, vertex_label=G_person.vs['person'])

### Comparison of centralities

Last but not least, let's compare the centrality of suspicious characters in the network. Paul Revere is way on top.

annotations = nodes['person'].to_list()
x = nodes['eg']
y = y=nodes['betweenness']
plt.scatter(x, y)
for i, label in enumerate(annotations):
    plt.annotate(label, (x[i], y[i]))
plt.show()