# Collaboration Networks in the US Congress

In this project, we will work through a real world example looking at collaboration networks in the US Congress. It is an advanced example, but I nevertheless wanted to show you how you can work through a real dataset. It is typical in the sense in terms of the data challenges you face with social data as well as the kind of new questions you can ask.

Legislatures are inherently difficult to investigate. Political scientists have been interested in collaborations among parliamentarians as explanatory factors in legislative behavior for decades. But up to now they have lacked the data to do this on a larger scale. Recently, legislatures around the world have begun to to publish all their workings on the web. Especially advanced is the US Congress. Take a look at  https://www.congress.gov/. 

In this exercise, we will first download a sample of bills and then investigate sponsors and cosponsors of these bills. If you do not know much about the workings of the US congress, take a look at https://en.wikipedia.org/wiki/Sponsor_(legislative) 

First we load a couple of libraries that we need. Some you will recognise. Hopefully, all the libraries should have been installed for you beforehand if you use a framework like Anaconda (https://www.anaconda.com/). But if at any moment in time Python does not recognise the library, you can simply install it by following the instructions at https://www.codecademy.com/articles/install-python.

#Run the code below

import re
import requests
import random
import json

from random import randint
from time import sleep

from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

Next we define the congress number we are interested in. Because we started with it, I define it next as the 114th congress but you can change it of course. Just make sure that the congress actually exists.

#Run the code below

congress_no = 114

Our aim is to download a random sample of bill information for this congress. The first challenge is to find out how many legilsations they were in this congress. 

We have defined a URL_all for you where you can find the information with some web parsing. How?

#Run the code below

URL_all = 'https://www.congress.gov/search?searchResultViewType=expanded&pageSort=documentNumber%3Adesc&q=%7B%22source%22%3A%22legislation%22%2C%22congress%22%3A' + str(congress_no) + '%7D'

Now do the web parsing to find the number of legislations and assign the result to max_legislation. Finally, print it out. 

# How many legislations?
# Use beautiful soup, lecture video 2 (4 min)
# page = requests.get(url)
# soup = beautifulsoup(page.text, 'html.parser')
# use re.findall (regular expressions)

Create a sample of bills you would like to download. max_n is that number.

#Run the code below

max_n = 10

Create random_bills as a random list of integers with max_n bill numbers you can download.

# random.sample max_legislation

Download the random_bill texts into a list called pages that is defined for you. Make sure that the bill page exists in case you happen to have selected a bill number that does not (status_code = 200). After accessing each bill, I recommend to put Python to sleep for a little bit in order for the congress web server to have a chance to recover. Once you have run this, do not repeat this too often to not overload the server.

The information you are looking for is in the 'all-info' pages for a bill. So, e.g., https://www.congress.gov/bill/114th-congress/house-bill/5245/all-info. Here, you can find later on the sponsor and co-sponsor information.

Tip: You want to use the requests library to download the all-info pages.
Second tip: This is the moment a progress bar in your for loop is very handy. Check out https://github.com/tqdm/tqdm.



We haven't spoken about this yet, but the most commonly used library in Python to parse websites is BeautifulSoup. Check out https://realpython.com/python-web-scraping-practical-introduction/.

BeaurifulSoup generally takes in a page from requests and applies an HTML parser. We have already loaded its class from the bs4 library above.

Next define a function extract_data_website that takes in the result of BeautifulSoup parsing of a website called soup, a bill number and for that bill returns a dataframe with the bill number, its sponsors and its co-sponsors. 



Next run extract_data_website agains all pages in the pages list and build up a main dataframe called sponsors_bill_df.



Print out the first couple of rows of sponsors_bill_df.



Create two lists of all sponsors and cosponsors as well as a list of all bills. These lists should be called all_sponsors_cosponsors_list and all_bills_list and should be sorted alphabetically.





A dataframe is nice but for a numerical analysis like clustering we often need a matrix. Run the next cell to create an empty sponsors_cosponsors_matrix.

#Run the code below

sponsors_cosponsors_matrix = pd.DataFrame(np.nan, index=all_bills_list, columns=all_sponsors_cosponsors_list)
sponsors_cosponsors_matrix.head(1)

Now fill this matrix. The matrix should contain 'Sponsor' for the sponsor of a bill and 'Cosponsor' for the cosponsors. Otherwise, all cells should have NaN values.



Print out the first couple of rows of this very sparse matrix.



As this matrix describes a community of sponsors and cosponsors we will try our clustering approach next. We try k-means on clusters of sponsors and cosponsors.

## Kmeans

First we load the library and define k = 5.

#Run the code below

from sklearn.cluster import KMeans

k = 5                      

We need to do some data preparation next. Define a matrix kmeans_matrix that is 1 where sponsors_cosponsors_matrix contains a sponsor or cosponsor and otherwise 0.



Run the following cell for the k-means clustering.

#Run the code below

kmeans_clusters = KMeans(n_clusters  = k) 
kmeans_clusters.fit(kmeans_matrix)

Print out the cluster centres for all members. 



Something intereting might be hidden in all this but overall the matrix does not have enough entries to deliver any meaningful conclusions from clustering.

## Apriori

This is why we would like to try another approach. We will try to learn rules that represent the data. We imagine that it should be possible this way to determine rules like if senator x proposes a bill, congresswoman y will be involved, too, as she is associated with him. 

Association rules are often used for 'market basket analysis', which are typical example for analysing very sparse datasets. The result of a market basket analysis is a set of association rules that specify patterns of relationships among items. A typical rule might be expressed in the form: 

- {peanut butter, jelly} -> {bread}. 

So, if a costumer buys peanut butter and jelly, she will also get bread. We will try to learn something similar by associating US congress members who sponsor bills together. If congresswoman X sponsors a bill, so will senator Y.

Perhaps the most-widely used approach for efficiently analysing large data for rules is known as Apriori. This algorithm was introduced in 1994 by R. Agrawal and R. Srikant, and has become somewhat synonymous with association rule learning since then. Check out: Fast algorithms for mining association rule, in Proceedings of the 20th International Conference on Very Large Databases, pp. 487-499, by R. Agrawal, and R.Srikant, (1994).

However, I found that the apriori algorithm (https://pbpython.com/market-basket-analysis.html) is extremely slow in Python. So, rather than Apriori we will use a faster version (http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/) that implements Han, Jiawei, Jian Pei, Yiwen Yin, and Runying Mao. "Mining frequent patterns without candidate generation. "A frequent-pattern tree approach." Data mining and knowledge discovery 8, no. 1 (2004): 53-87. It concentrates on frequent itemsets.

"FP-Growth is an algorithm for extracting frequent itemsets with applications in association rule learning that emerged as a popular alternative to the established Apriori algorighm. In general, the algorithm has been designed to operate on databases containing transactions, such as purchases by customers of a store. An itemset is considered as 'frequent' if it meets a user-specified support threshold. For instance, if the support threshold is set to 0.5 (50%), a frequent itemset is defined as a set of items that occur together in at least 50% of all transactions in the database."

We follow the examples given at http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/ including the TransactionEncoder. But we could have actually reduced the steps, as we have already have sponsors_cosponsors_matrix, as we will later see. 

Load the Python version of FP-Growth by running the cell below. You might have to install mlxtend first, a library of data science tools. See http://rasbt.github.io/mlxtend/installation/.

#Run the code below

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

The fpgrowth library we are going to use requires our dataset to be in the form of a list of lists, where the whole dataset is a big list and each transaction in the dataset is an inner list within the outer big list. Currently, we have data in the form of a pandas dataframe. 

Convert sponsors_bill_df into such a list of list called records. The inner list contains for each bill the list of sponsors and cosponsors. The outer list is simply all the bills.



Follow http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/, to transfrom records into the right format for fpgrowth via the TransactionEncoder.



Do you see what we could have done directly with sponsors_cosponsors_matrix in order to arrive at the same output? All the association mining/clustering algorithms have similar inputs.

Let us finally return the records with at least 20% support. 20% is not much of course, but we also limited ourselves to only a small subset of bills.

Tip: Make sure to set use_colnames=True to get a readable output.



If you want you can download more bills and play with the support. 

Or you could now start to look up the sponsors/co-sponsors on wikipedia and find out whether there are, e.g., bi-partisan bills?

## Network

We have learned a lot today about how to analyse communties with algorithms. While k-means was less convincing because there was too little data, the association rules migth show a few interesting results at least?

Most people would look at another way of working through this data with so-called social network analysis. Check out https://en.wikipedia.org/wiki/Social_network_analysis. 

In Python we need the networkx libraries.

#Run the code below

import networkx as nx

With the command below you create a social network graph G from sponsors_bill_df.

#Run the code below

G = nx.from_pandas_edgelist(sponsors_bill_df, "Sponsor", "Cosponsor", ["Bill"])

Now, try and visualise the network G. Check the documentation if you need to. 

There are a lot of online tutorials how to use networkx. The hard part will be plot all the nodes and edges in the graph so that they make sense together.



If you want, you can make the network as informative and complicated as you want ... Let's move on for now to a more hands-on analysis.

After all this association mining with mixed results, you might decide to go another way and look at the US Congress bios for further review. Maybe, you want to know more about the biographical background of members of congress or mine their Twitter feeds. We can access the propublica congress API for that. It is explained at https://www.propublica.org/datastore/api/propublica-congress-api.

First define the correct url.

#Run the code below

propublica_url = 'https://api.propublica.org/congress/v1/' + str(congress_no) + '/senate/members.json'

Now request an API key from https://www.propublica.org/datastore/api/propublica-congress-api and use it to set 'X-API-Key' in a 'headers' dictionary for requests.



Fetch the data and print out the first results in number of characters. I used 3000 characters, but feel free to print out as many as you want.



It is a bit difficult to read but you hopefully recognise this as a JSON response, which is very typical for API calls. 

You can read the results into a dictionary with the json package that we imported earlier.

Load this dictionary into a a dataframe called members_df and print out the first couple of rows. It's very easy to do with the right pandas function.



Print out the column names of members_df.



With members_df and the column names, create two functions: 

The first function 'get_representative_bio' takes in a congress number and returns a dataframe of people in the house of representatives that shows the following information for them: ['first_name', 'last_name', 'date_of_birth', 'gender', 'party', 'state', 'next_election', 'missed_votes_pct', 'votes_with_party_pct'].

The second function 'get_senator_bio' does the same but for the senate. 





Print out the first results for get_senator_bio(114)



I hope you see how rich the information is datasets linked to the US Congress.

Association mining and clustering are very powerful techniques to find yet unknown relations. We tried them to detect patterns of maybe unexpected behaviour in the US Congress. In another example, we used them to detect how powerful the big Internet companies are in the mobile ecosystems: https://journals.sagepub.com/doi/full/10.1177/2056305120971632

"Our research seeks to understand how platforms have been able to technically integrate themselves into the fabric of the mobile ecosystem, transforming the economic dynamics that allow these largely enclosed entities to compete. We therefore want to consider platforms as service assemblages to account for the material ways in which they have decomposed and recomposed themselves for developers, enabling them to shift the economic dynamics of competition and monopolization in their favor. This article will argue that this shift in the formation of platform monopolies is being brought about by the decentralization of these services, leading to an overall technical integration of the largest digital platform such as Facebook and Google into the source code of almost all apps."


