# Detecting Political Communities of Practice

## The US Congress


Today, we start exploring political and social community data using a popular exploration technique with association mining like clustering. It is very effective and easy to use. We will gain some real insights about US politics as well as communities in social networking sites.

Another focus today will be to introduce you to the problems of real life social datasets, which often lack the quality to be processed easily. For instance, you will learn various strategies of dealing with missing entries either by removing them completely or by trying to replicate its values. The kind of social data we are dealing with is vast and unorganized, which makes organizing it for analysis no easy task. In reality, you will spend most of your time on working through such data challenges. 

Finally, today will be dedicated to data exploration and the insights you can gain here. Exploring data is not necessarily a very structured part of your work. 

I hope you all know Python's Pandas framework. If not, you might want to check out the online documation or read a good book. Please, ask me in class.

![title](../img-videos/us-congress.jpg)

You might remember the power of Pandas functions like head(), describe(), etc. to quickly explore essential components of data. Some people consider data exploration to be the most important part in the data analysis process, as it is really important to understand each aspect of the data and the way it is represented. In social and cultural analytics, most of our work is based on data exploration techniques rather than prediction. We will cover prediction later in the course. 

First load all the standard Pandas, etc. Python libraries, which should be familiar to you.

#Run the code below

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

## Clustering 

In this section, we concentrate on introducing the power of digital methodologies and data exploration using a particular method called clustering, which is closely related to the understanding of political and social communities. We will look at the basics of clustering that delivers you powerful results very fast. In particular, we will use the k-means algorithm, which was invented by MacQueen in the late 1960s (https://en.wikipedia.org/wiki/K-means_clustering). Despite being ancient in terms of computer lives, k-means is still widely used as it delivers good results with great performance. Performance in computing describes the effort we need in terms of computational resources. You will generally notice performance by the execution time. 

https://machinelearningmastery.com/clustering-algorithms-with-python/ provides a good overview of clustering algorithms that are implemented in Python.

Next you see the first of three presentations in this session, which introduce you to the background. Run the next cell to activate it and enjoy ... 

#Run the code below

from IPython.display import Video

Video("../img-videos/Session1.mp4", embed=True)

As discussed in the video, k-means tries to develop clusters by 
(1) initialising a pre-defined number (k) of randomly chosen centroids in space. Centroids are simply the centre points of clusters. 
(2) The algorithm assigns each observation to the cluster with the closest centroid. 
(3) Based on how balanced this assignment is, the centroids are recalculated and steps 1 and 2 are repeated until the algorithm balances out. 

Let's move on to some actual work.

In a first exercise, we will use k-means to understand voting behaviour in the US senate. We selected a senate that was not too partisian as we would like to investigate changing voting behaviour. 

Please, run the cell below to create the congress_114 data frame, which contains the voting behaviour of 114th US Senate. According to Wikipedia (https://en.wikipedia.org/wiki/114th_United_States_Congress), the 114th Congress met in Washington, D.C. from 3 January 2015 to 3 January 2017, during the final two years of Barack Obama's presidency. 

The 2014 elections gave the Republicans control of the Senate (and control of both houses of Congress) for the first time since the 109th Congress. With 247 seats in the House of Representatives and 54 seats in the Senate, this Congress began with the largest Republican majority since the 71st Congress of 1929–1931. There are 23 Democrats, 1 Independent and 33 Republicans in our dataset. Please note that this does not represent the full 114th congress but a sample. 

#Run the code below

congress_114 = pd.read_csv("data/114_congress.csv")

To warm up check the first five entries of the dataset. It contains the name of a particular senator, his/her party and home state as well as for each bill whether the senator voted for the bill (1) or against it (0). 



![title](../img-videos/us-senate-voting.jpg)

Next check the last five records. 

You will see that the last record contains lots of NaN values, which stand in Python for missing values. This is the voting record of a senator who was not able to vote. In real-life datasets, you will see quite a few of these kinds of records – maybe because they never existed or they were not recorded in the first place, etc.



There are many strategies to deal with these kinds of missing records or 'dirty' data. Here, we will use the brute-force version and simply remove it from the data set. It is only one record and is completely missing. So, removing these records should be safe.

First check that there is really only one record by displaying all null entries in the dataset.



That's only one record missing, and the person has really not voted at all. This makes us confident that we can just delete them ...

Remove the record. The easiest is to simply remove all the records with NaN values. Pandas has a function for that, which you should know by heart.



Check the last 5 elements again to make sure that the observation with NaN values (of the senator who missed votes) is really gone.



Try to get a quick overview of the dataset and 'describe' it.



This has not produced very useful outputs. Check the column types to undestand why ...



Finally, check how many democrats, republicans and independents there are in congress_114. As you can see this is only a subset of the 100 senators. 



We want to improve the content of the dataframe next and make the types fit better. To this end, we change the float types to integers. 

This is actually quite hard work in things like Pandas. You need to first select the right columns in the data frame. You could just count the column numbers, as it is a very small data frame. But there is a trick to find the indexes automatically. We match the beginnings of the column names and are only interested in those that start with "bill-". 

Check first the names of the columns to verify that there is a pattern you can exploit.



Next, find a string function to select only the 'bills-' columns. Assign them to bill_cols.



With the Pandas loc function and bill_cols, you can select all the bill columns. Make them all integer columns with astype(int). 



Check that everything has come out as planned by running dtypes. 

BTW, string types are objects in Pandas ...



## Clustering with K-means

Many of the decisions in analytics are more an art than a science. We need to often estimate many parameters – either based on previous experience or using background knowledge. K-means is famous for heavily depending on k or the number of clusters we want to assign. We need to tell Python which k to use. 

In order to find a good starting point for k, we can use our own knowledge about how the US senate is structured. We would like to investigate voting clusters, and we know that the US senate is dominated by 2 major parties. So, it seems like a good idea to start with two clusters (k = 2), as we can assume that there should be two major party-based voting clusters. Please, assign k = 2.

#Run the code below

k = 2

Next, we need to understand what we would like to cluster and choose the relevant features as input into the k-means algorithm. If you look back into your earlier explorations of the dataset, you can see that the first 4 columns do not contain voting behaviour. They have the name, state, etc. of the various senators. The voting behaviour can be found in columns 5 to 19. Use either the column indexes or bill_cols to create a new dataframe congress_114_voting, which only contains the voting behaviour.



Great, we are ready to cluster the votes. Check out the details of k-means in the SKlearn documentation. 

Its main arguments are the dataset to cluster and the number of clusters. We can leave all the other inputs at their defaults. 

First import KMeans from sklearn.cluster.

#Run the code below

from sklearn.cluster import KMeans

Now, run KMeans and fit it with n_clusters  = k. Check the documentation ...



This should not have taken too long, as the dataset is very small. 

In the meantime, you can celebrate that you have just run a machine learning algorithm. k-means is a fairly simply one, but still a standard example of an unsupervised machine learning algorithm. Unsupervised machine learning means that you do not have to train the computer in advance about the kind of results you expect. You can also check out what that means in a wonderful Coursera/Stanford course under https://www.coursera.org/learn/machine-learning/lecture/olRZo/unsupervised-learning. The course is legendary and gives you in-depth knowledge of machine learning. 

Check why this was so quick by printing the number of iterations required to converge. Check the documentation ...



Finally, the cluster assignments are stored as a one-dimensional NumPy array in kmeans.labels_. Here’s a look at the first five predicted labels.

#Run the code below

kmeans.labels_[:5]

Ok, so now that we have run our first machine learning algorithms, what do we do with the results? A good first step for k-means and other clustering algorithms is to check out the size of the clusters. Who do you expect to belong to each cluster?

Use np.bincount with kmeans.labels_, please.



These numbers show that there is not really a clear division between Republicans and Democrats, as the clusters do not correspond to the numbers each party has in the senate. 

Create a new dataframe congress_114_result, which contains the first 4 columns of congress_114 as well as the cluster assignments for each senator by kmeans.

Take a moment to reflect what we gain from such a new dataframe?



Because we like it tidy, we give the columns of congress_114_result new readable names. Remember that we can do this by assigning the columns directly to a list of names.

#Run the code below

congress_114_result.columns = ['index','name','party', 'state', 'cluster']

Let's take a look at the composition of  congress_114_result. This time we want to take a look at the whole congress_114_result. What do you see?



Finally, let's take a look at the composition of our 2 clusters. 

In this case, we want to count how often a Democrat appears in cluster 1 or how often in cluster 2; similarly, how often is a Republican part of either cluster 1 or 2. Please note that there are also Independent senators. 

In order to compare party and cluster features, use pd.crosstab. Remember that with the crosstab function we can count the frequency of the combination of two columns. 



Take a minute to interpret the results. Which party is more coherent in its voting behaviour? Can you identify the outliers by looking through the result data frame? 

k = 2 seems to have been a fairly good choice as there is a lot of overlap between parties and vooting clusters. 

Let’s try k = 5 next to get more diversified results with 5 clusters. 

#Run the code below

k = 5

Fit Kmeans with the new k. Assign it to kmeans_5.



As we already have a congress_114_result data frame, we just need to create a new column in it with the new data. Create a new column cluster_5.



Let's run congress_114_result to check the results.



Now, let’s compare voting behaviour and parties again with crosstab.



There is a strange outlier in the clusters with regard to voting behaviour of particular Republicans. Let’s investigate those Republicans who appear not to vote with the rest of their party or other Democrats. 

The cluster number will change depending on the result of your k-means. So, anything between 1 and 5. You get this number from the table you have just printed out. Use it to filter congress_114_result and retrieve the names and states of the senators. 



### Visualisation

Finally we want to also visualise the cluster assignment to present how senators are close to each other in 2-dimensional coordinate system.

Create a simple visualisation that maps the 5 clusters in 2 dimensions and colour the points that represent the senators according to their 5 kmeans clusters.

Tip: You need the principal component analysis trick to map congress_114_voting into 2 dimensions.

First load PCA from sklearn.

#Run the code below

from sklearn.decomposition import PCA

Now apply a PCA with two components. 

Then, create a new dataframe principal_df with the results of this analysis and name the two columns PC_1 and PC2.

Finally, add the columns name, party, cluster and cluster_5 to principal_df



Run the code below to visualise. Do you understand what it is doing?

#Run the code below
#based on https://honingds.com/blog/seaborn-scatterplot/

import seaborn as sns
plt.figure(figsize=(15,7))

sns.scatterplot(data = principal_df, x='PC_1', y='PC_2', hue='cluster_5', 
                style='party', s=100, palette=['green','grey','red','blue','orange'], alpha=.40)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))

label_point(principal_df.PC_1, principal_df.PC_2, principal_df.name, plt.gca()) 


Take some time now and investigate which senators are the outliers are by researching them online. WIkipedia is enough. Can you understand why they are clusters by themselves? 

### Determine the optimal number of parties in the Senate according to K-means

So what would be the optimal number of parties in the senate? In terms of k-means, we need to determine the optimal k. There are quite a few methods to estimate k. Among the best known is the elbow method that is based on visualing trials with several k's.

The elbow method is a useful graphical tool to estimate the optimal number of clusters k for a given task. Intuitively, we can say that, if k increases, the within-cluster 'Sum of Squared Errors' (SSE; also called 'distortion') will decrease. 

The idea behind the elbow method is to identify the value of k where the distortion begins to decrease most rapidly, which will become clearer if we plot the distortion for different values of k.

To perform the elbow method, run several k-means, increment k with each iteration until you reach max_k = 10, and record the SSE score in a list called sse. Then, map the SSE for each iteration to find the point for curve bends, the elbow. This will be the best k.



Run the code below to visualise the elbow ...

#Run the code below

sns.set(style='darkgrid')

elbow_df  = pd.DataFrame(list(zip(range(1, max_k), sse)),
              columns = ['Number of Clusters', 'SSE'])


sns.lineplot(x = 'Number of Clusters', y = 'SSE', data = elbow_df)

As you can see 2 is already the best answer ...

You can now continue playing with different k values if you want. 

Before we move on to the world of teens, please finally consider https://www.r-bloggers.com/k-means-clustering-is-not-a-free-lunch/. The article describes that, while clustering (and other machine learning algorithms) can produce very persuasive results, these do not come for free. They are no free lunch. The results always depend on the assumptions we put in such as the number of clusters in k-means but also how we describe the vote in Congress, how we measure somebody’s influence, etc. This is the famous 'No Free Lunch Theorem' in Machine Learning. 

