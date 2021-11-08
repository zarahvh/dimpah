# Social Data from the Web

## Social Networking Communities

We have just discussed how clustering can be an effective tool to understand political behaviour. As an unsupervised learning technique it provides a new machine reading on party affiliations. Another popular application of clustering is detecting communities in social relationships.  Next we go through an example and dataset in Brett Lantz’s excellent book on Machine Learning (Lantz, B. (2013). Packt Publishing Ltd.). The dataset is discussed on pp. 279. It covers the relationships in a Social Networking Service (SNS). While this is a fairly early SNS dataset, it is freely available and offers similar kind of insights you can gain from my recent examples. 

This section also introduces you to the intersection of digital marketing techniques and sociological studies of online networks.

Lantz explains that the dataset was compiled for sociological research on teenage identities at Notre Dame University. It represents a random sample of 30,000 US high school students who had profiles on a well-known SNS in 2006. At the time the data was collected, the SNS was a popular web destination for U.S. teenagers. Therefore, it is reasonable to assume that the profiles represent a fairly wide cross section of American adolescents in 2006. The data was sampled evenly across four high school graduation years (2006 through 2009) representing the senior, junior, second-year and freshman classes at the time of data collection. Then, the full texts of the SNS profiles were downloaded. Each teen's gender, age and number of SNS friends was recorded. 

A text-mining tool was used to divide the remaining SNS page content into words. From the top 500 words appearing across all pages, 36 words were chosen to represent five categories of interests, namely extracurricular activities, fashion, religion, romance and antisocial behaviour. The 36 words include terms such as football, sexy, kissed, bible, shopping, death, drugs, etc. The final dataset indicates, for each person, how many times each word appeared in the person's SNS profile. 

First we load the relevant packages and the dataset. Run the cell below

#Keep cell

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline


teens = pd.read_csv("data/snsdata.csv")

Print out the first couple of rows from the teens dataset. You know how ...

teens.head()

The teens dataset is now loaded into your environment. Take a close look and make sure you understand how it is produced.

We can use the info() method to output some general information about the dataframe. Run `teens.info()`.

teens.info()

Part of this lesson is centered on the issue of looking into real-life data on digital society. We have mentioned earlier that a common problem is that observations/records are missing in such data, which is indicated by the NaN value in Python - as you might remember. 

In the info printout, you can also see that the non-null count is lower for those columns that contain NaN values. Gender is one of them.

Let's watch a quick video first how to deal with dirty data in general.

<video width="90%" height="90%" controls src="img-videos/Session2.mp4" />

That was a lot of information. Let's go through this one step a time with the teens data.

Print out the absolute non-null value counts for gender as well as the relative ones with:
```
print(teens['gender'].value_counts())
print(teens['gender'].value_counts(normalize=True))
```

print(teens['gender'].value_counts())
print(teens['gender'].value_counts(normalize=True))

With dropna set to False in value_counts, we can also see NaN index values. Try that ...

teens['gender'].value_counts(dropna=False)

But missing values are not our only problem. At least as common are misreported observations in real-life data. As an example, let’s look at the at the age distribution of the teens' age. You can do this in several ways but you should always print out maximum and minimum values. Run: `teens['age'].describe()`.

teens['age'].describe()

There are quite a few strange records here. Teens can have a minimum age of less than 4 and a maximum age of over 100! These cannot be considered teenagers. 

As a rule of thumb, let’s assume teenagers are between 13 and 19 years old. Let’s mark all other teens' age entries as invalid. We say that invalid entries should have a NaN value. Set this, please, by typing in `teens.loc[(teens['age'] < 13) | (teens['age'] >= 20), 'age'] = np.nan`.

teens.loc[(teens['age'] < 13) | (teens['age'] >= 20), 'age'] = np.nan

The next step in our data cleaning process is to replace NaN values. Of course, we could simply remove all rows/observations, for which we have null entries. We did this effectively in the senate example. But then there was only one row that contained null values. In the SNS example, we would lose too many rows with such a brute-force approach. So, let’s try and fill the null values with estimated values. 

Let’s start with the gender and replace null values by creating new columns for male and females. 

To this end, we first create a new column to record all the female teenagers. Create a new column 'female' that is set to 1 if the teenager is female and 0 otherwise. Run `teens.loc[(teens['gender'] == 'F') & (teens['gender'].notna()), 'female'] = 1` to set the female column to 1 for females. notna() selects all rows that are not NaN.

teens.loc[(teens['gender'] == 'F') & (teens['gender'].notna()), 'female'] = 1


Can you set the female column to 0 for males? You just need to change one letter ...

teens.loc[(teens['gender'] == 'M') & (teens['gender'].notna()), 'female'] = 0

Next we will create another column for the null values in gender we want to call no_gender. Set this to 1 if there is no gender recorded and otherwise to 0. 

This process is called dummy-coding btw. This is typical to community analysis. A dummy variable is a numerical variable used in the analysis to represent subgroups – in our case males, females and others. In research design, a dummy variable is often used to distinguish different groups to address them differently. By creating a separate column per gender entry, we can compute clusters for separate gender communities. 

Check out dummy-coding on the web. Can you find easier ways to do this in Pandas?

Run 
```
teens.loc[teens['gender'].notna(), 'no_gender'] = 0
teens.loc[teens['gender'].isna(), 'no_gender'] = 1
```

teens.loc[teens['gender'].notna(), 'no_gender'] = 0
teens.loc[teens['gender'].isna(), 'no_gender'] = 1

After this, we have the original column, a new column called female, which contains information about whether the teen is female or not (male) and a new column with information about whether the gender value is missing. Using this column we could, for instance, check with clustering whether certain communities have a tendency not to record their gender values. How? 

Check out the changes with teens.head(). You have to scroll all the way to the right to find the new columns.

teens.head(20)

It's very easy now to calculate the number of teenagers where we do not have gender entries for. How? Remember sum()?

teens['no_gender'].sum()

Did you find that there are 2724.

The age column is next. 

Can you find the average age and take care that null values are discounted? Tip: run Pandas' mean and set skipna = True: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html

teens['age'].mean(skipna = True)

What would happen if you set skipna to False?

A good strategy to overwrite missing age values would be to use the average age value and assign it to all of the missing ones. This process is called mean-imputation and is employed frequently. Pandas has some real strengths here. Check out https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html.

Pandas makes you life very easy with its fillna function. Run the following cell.

#Keep cell
teens['age'].fillna(teens['age'].mean()).quantile([.25, .5, .75])

Let's further improve this with some good old-fashioned human intelligence.

We feel confident that we can do better than just the mean, because we know the graduation year, too. This is the year our teens are supposed to graduate. It seems a reasonable assumption that those teenagers with an earlier graduation year should be older than those for whom graduation is further away. 

We can easily find this out by running the mean function for each graduation year group separately. Type in `teens[['gradyear', 'age']].groupby(['gradyear']).mean()`. 

teens[['gradyear', 'age']].groupby(['gradyear']).mean()

Let's take a moment and look at groupby() as it is essential knowledge to deal with Pandas: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html. It is at the heart of the split-apply-combine paradigm that will keep us busy for the rest of this session: https://pandas.pydata.org/docs/user_guide/groupby.html. Take a close read and you will see that we will cover all its elements throughout the session today. Groupby allows you to to 'split' data into distinct groups. This is often done with the intention to then 'apply' functions afterwards like in our case mean(). This works amazingly well but requires lots of practice in my exeperience. So, let's move on.

According to our last output, our suspicion has proven right. There is a significant difference in the average ages depending on the graduation year. Let’s use this knowledge and update missing values in the age group depending on the graduation year. To this end, you actually have to do a lot of Pandas labour, which demonstrates that 80% of the work of a data scientist lies in working with data. But I am sure you know this by now.

You can, e.g., proceed with the following strategy:
Create first a temporary dataset with the results from the above group_by call. Then merge this new dataset with teens and replace the null values of teens.age with the ones from the temporary dataset.

Create the temporary dataset first with `ave_age = teens[['gradyear', 'age']].groupby(['gradyear'], as_index=False).mean()`. Also print out ave_age. It should be a data frame of three rows. 

Wondering about as_index=False? Check out https://pandas.pydata.org/docs/user_guide/groupby.html. We simply do not want to create a new index from the groupby argument gradyear.

ave_age = teens[['gradyear', 'age']].groupby(['gradyear'], as_index=False).mean()
ave_age

Update the teens age columns but make sure that in the end you have not added additional columns. First you need to merge teens and ave_age on gradyear. Run `teens = pd.merge(teens, ave_age, on=['gradyear'])`. Also print out the first few columns of teens.

Merge is another powerful command to learn about. It's part of the Merge, join, concatenate and compare group of command - some of which we have already met: https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html. If you happen to know database languages, you will know everything about it. Merge() combines two data frames on common columns - in our case gradyear.

teens = pd.merge(teens, ave_age, on=['gradyear'])
teens.head()

With some scrolling, you should now see two age columns. age_x is the original one from teens, while age_y is the estimated value based on the gradyear. Now, we want to replace the age_x (the original value) with age_y if age_x is NaN. Run `teens.loc[(teens['age_x'].isna()), 'age_x'] = teens['age_y']`.

teens.loc[(teens['age_x'].isna()), 'age_x'] = teens['age_y']

Now, we only need to so some cleaning up. Give age_x its old name age back and drop age_y, which we don't need anymore. Run the cell below.

#Keep cell
teens.rename(columns={'age_x': 'age'}, inplace=True)
teens.drop('age_y', axis=1, inplace=True)

Use `teens.info()` to see that the age column does not contain NaNs anymore.

teens.info()

Check out new ave_age with `teens['age'].mean()`.

teens['age'].mean()

This is all quite advanced stuff but as long as you remember the kind of steps we have taken you should be able to impute one column's missing values by using another column as a reference. In our case, we use our knowledge that age is dependent on gradyear to find the missing values. Please, take some time to review the steps.

Let’s take a look at the resulting age column with `teens['age'].describe()`.

teens['age'].describe()

This looks much better. We have now learned how to delete missing values completely or impute them using a background knowledge. 

After we have dealt with the missing records, I think we are ready to cluster again. We will use our trusted k-means without actually referring to either age nor gender. Sorry! But it was good that you learned how to deal with missing values and we will use them later. 

Just like in the US Senate example, we need to first understand, what we are trying to cluster. In the Senate example, we clustered voting behaviour. Now, it will be interests, which we can get from the columns 5 to 40 of the teens data frame. This time we select them by number as there is no clever way of selecting them by expression as for the Senate data.

Please create the interests dataframe by selecting columns 5 to 40 from teens with `interests = teens.iloc[:,5:40]`. Also, print out the first few rows of interests.

interests = teens.iloc[:,5:40]
interests.head()

We did not mention this before, because it was not necessary but k-means is very sensitive to input of varying size, length, etc. It was not necessary to focus on this in the previous example, because all the voting behaviour was in the range of 0 to 1. Now, we have interests of very different ranges. The interests are simply based on how many times a keyword appears in teenagers' social networking contributions.

Since k-means is based on calculating distances between data points and their centroids, it will be strongly influenced by the magnitudes of the variables we cluster. Think about if for a moment! Just imagine one column having values running from 1 to 10 and another from 1 to 1000. How could we compare distances between them? 

We therefore need to scale the value so that they all fall into the same range. To this end, Python has the scale function in scipy.stats, which centres values around their mean. Using the apply function, we can tell Python to scale all interests values. apply is an alternative way to loop over values in a column and apply a function: https://www.datacamp.com/community/tutorials/pandas-apply.

Run `from scipy.stats import zscore` to import zscore, which is very popular in data analysis for standardisation: https://www.statisticshowto.com/probability-and-statistics/z-score/

from scipy.stats import zscore

Apply zscore and assign the results to a new dataframe interests_z with `interests_z = interests.apply(zscore)`. Finally, print out the first few columns.

interests_z = interests.apply(zscore)
interests_z.head()

It's clearly normalized around the means of the various columns - by the distance of the standard deviation.

### Clustering

Now we cluster again and start by importing KMeans. Run the cell below.

#Keep cell

from sklearn.cluster import KMeans

We decide 5 clusters is enough and assign k = 5. Run the cell below.

#Keep cell

k = 5

Now we are ready to cluster. Create and fit teen_clusters the way you know. It is just a copy and paste job from before.

teen_clusters = KMeans(n_clusters  = k) 
teen_clusters.fit(interests_z)

Let’s investigate the size of the clusters with .labels_ and np.bincount.

np.bincount(teen_clusters.labels_)

I have noticed very different results depending on the kmeans results. I suggest to rerun kmeans a couple of times until you see a distribution that looks ok. You want to especially avoid clusters of only one 1 item. This would be the time to determine the optimal k with the elbow method, but we don't want to to that right now and move on.

We can also look at the centroids/centres with teen_clusters.cluster_centers_. You learned earlier how to pretty-print this in a data frame. Run:
```
interests_centroids = pd.DataFrame(teen_clusters.cluster_centers_, columns=interests_z.columns)
interests_centroids
```

interests_centroids = pd.DataFrame(teen_clusters.cluster_centers_, columns=interests_z.columns)
interests_centroids

A simple way to detect clusters is to find thr maximum values in the columns. Try it by using the idxmax() function from Pandas: `interests_centroids.idxmax()`. Check out its documentation.

interests_centroids.idxmax()

Hopefully, your results look similar to the table from Lantz (2013) on p. 288: 

![title](img-videos/teen-clusters.jpg)

Do the names of the clusters make sense to you? Do you remember all those teenage movies you watched?

Next, let’s continue with another type of analysis. Let’s first assign each teen to a cluster, as we did before in the voting example for the senators. Please, add a column called 'cluster' to the teen data frame with `teens['cluster'] = np.array(teen_clusters.labels_)`.

teens['cluster'] = np.array(teen_clusters.labels_)

Let's take a look at the teen data frame with head(). All the way to the right, you can find the cluster assignment.

teens.head()

Let's print out the first 5 teens and only the columns 'cluster', 'gender', 'age' and 'friends'. I hope you remember how this works. How do we select the first 5 rows? How do we select the columns? Tip: loc is the function you are looking for.

teens.loc[:4, ['cluster', 'gender', 'age', 'friends']]

Since we have learned earlier how to group by particular interests, let’s aggregate the teens' features using the clusters. 

Print out the average ages per cluster. Do you remember how this works? Tip: Replace gradyear with cluster and you are ready to go.

teens[['age', 'cluster']].groupby(['cluster'], as_index=False).mean()

The clusters do not differ in terms of ages very much. There is no immediate relation between age and interest clusters. Now, let’s look at the female contribution to each cluster. How? Which column to you have to use instead of age?

teens[['female', 'cluster']].groupby(['cluster'], as_index=False).mean()

Overall, 74 per cent of the SNS's users are female. That’s why they contribute so much to each cluster. Can you see the cluster that has the most female users? Do you know why? Look back to the earlier analysis of the interests linked to the clusters ...

You can check for the average number of friends per cluster now. Just define the target of the aggregation per cluster as friends instead of female or age in the expressions above.

teens[['friends', 'cluster']].groupby(['cluster'], as_index=False).mean()

Here, the differences are stronger. We suspect that the number of friends played a key role in assigning the clusters. That’s the nature of a social network, I guess. 

We have now completed our exercises on clustering and understanding political and social communities. For today, we have just one more important question to answer. How do we get access to the kind of data we worked on today? The teens dataset stemmed from a research project in sociology published online, while the US Senate voting behaviour was downloaded from US government websites. 

## Web Scraping

The web has become a unique source of data for social analysis. Munzert et al. (2014) in their book on 'Automated data collection (...). A practical guide to web scraping and text mining' (John Wiley & Sons) emphasize in the Introduction that 'the rapid growth of the World Wide Web over the past two decades tremendously changed the way we share, collect, and publish data. Firms, public institutions, and private users provide every imaginable type of information and new channels of communication generate vast amounts of data on human behavior. What was once a fundamental problem for the social sciences — the scarcity and inaccessibility of observations — is quickly turning into an abundance of data. This turn of events does not come without problems. (…), traditional techniques for collecting and analyzing data may no longer suffice to overcome the tangled masses of data.' (p. XV). 

In short, we can find lots of data on the web. A big problem with web data is, however, that it is often inconsistent and heterogeneous. To get access to it, one often has to visit multiple web sites and assemble their data together. Finally, the data is generally published without reuse in mind, which implies that the data can be of low quality. That said, the web is so vast that it still provides an often overwhelming source of exciting data. 

Let's take a look at how we can access web data in general by scraping web sites.

<video width="90%" height="90%" controls src="img-videos/Session3.mp4" />

Returning to our first example of political communities, we will scrape data on the current composition of the US Senate from Wikipedia. 

This can be a complex task and requires additional libraries. But in this case, we can rely on Pandas directly with its read_html function that does all the hard work for you. Check it at https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_html.html.

All the content on the web is presented to us in a language called HyperText Markup Language (HTML; https://en.wikipedia.org/wiki/HTML). HTML is of course a way of presenting content on the web in a universal way. It also contains so-called hyperlinks that let you jump from web content to web content. 

If you are interested in the further details of HTML, why not take some time now to visit the excellent http://www.w3schools.com/html/, which contains a lot of practical exercises to learn everything about HTML and other web technologies. 

Each document on the web is identified by a URL. We set the url to the wikipedia page of current US senators and run the below cell.

#Run the code below

url = 'https://en.wikipedia.org/wiki/Current_members_of_the_United_States_Senate'



With the read_html function of Pandas, we can read in the web content behind the URL. However, if you check the documentation you need to provide Pandas with further details.

Unfortunately, web content in HTML format is not very structured and often simply chaotic. We would like to download only the table of the page of current US Senators and need to find a so-called 'match' for read_html to choose that table from the HTML document. 

Fortunately, for us there are many existing strategies to determine exactly the HTML element we would like to select. I looked for specific names in the table that are not repeated in the rest of the wiki page. Run: `senator_wiki = pd.read_html('https://en.wikipedia.org/wiki/List_of_current_United_States_senators', match = 'Richard Shelby')`.

senator_wiki = pd.read_html('https://en.wikipedia.org/wiki/List_of_current_United_States_senators', match = 'Richard Shelby')

If it all worked as it should run the cell below to create our senators dataframe.

#Keep cell
senators = senator_wiki[0]
senators.head()

Fortunately, the data is already of fairly good quality, but we still need to clean the data a bit.

Let's do some basic cleaning, where we ignore the strange textual errors and focus on the various columns that require direct attention. Please:

(1) Create a 'Party' column from whatever name read_html has given that column. In my case, it was called Party.1

senators['Party'] = senators['Party.1']

(2) Make sure that the 'Term up' is of type int.

Transfrom the column 'Term up' into an integer column with `senators['Term up'] = senators['Term up'].astype(int)`.

senators['Term up'] = senators['Term up'].astype(int)

(3) Clean the column 'Born' to only contain the numerical age and rename it to 'Age'.

As you can see the column Born contains the age of the senator, which we would like to extract from the string. As far as we can see these are the two digits in an otherwise string of letters. So, in (age 87) it is 87. To extract the 87, we can use regular expressions with Pandas. Take a look at https://www.dataquest.io/blog/regular-expressions-data-scientists/. We need the str.extract function https://pandas.pydata.org/docs/reference/api/pandas.Series.str.extract.html, which is very powerful. Run `senators['Age'] = senators['Born'].str.extract(r'.*(\d\d)')`.

The expression with extract says to read (r) all characters in the string (.*) and look for two digits (\d\d) to return these. Regular expressions require some practice and trial and error in my experience.

senators['Age'] = senators['Born'].str.extract(r'.*(\d\d)')

(4) Create a 'Years in Office' column that uses the information in 'Assumed office' to calculate how long the senator has served. Make sure that this column is of type int.

We first need to know which year we are currently in. We will use this to calculate the years left in office. We can use the datetime library and its now() function. Run the cell below.

#Keep cell
import datetime
year_ = datetime.datetime.now().year
year_

The years in the office of a senator can be calculated by subtracting fro year_ the year when the office was assumed. We can use our regular expression knowledge and simply look for the strings which are four consecutive numbers (\d\d\d\d) and return those. Type in `senators['Years in Office'] = year_ - senators['Assumed office'].str.extract(r'.*,.*(\d\d\d\d).*').astype(int)`. Observe that we use astype(int) to make the extracted string into an int.

senators['Years in Office'] = year_ - senators['Assumed office'].str.extract(r'.*,.*(\d\d\d\d).*').astype(int)

import datetime
year_ = datetime.datetime.now().year

senators['Party'] = senators['Party.1']
senators['Age'] = senators['Born'].str.extract(r'.*(\d\d)')
senators['Years in Office'] = year_ - senators['Assumed office'].str.extract(r'.*,.*(\d\d\d\d).*').astype(int)
senators['Term up'] = senators['Term up'].astype(int)


Finally, let's delete all unnecessary columns that you now changed such as 'Born' and 'Party.1'. Run the cell below.

#Keep cell
senators.drop(['Party.1', 'Born'], 1, inplace = True)
senators.head()

Just like before, we now would like to ask questions against the dataset and explore it. 

In particular, we would like to understand the pressure on parties during the next election for the US Senate. At the time of writing, these were the 2022 elections for Congress. We could now reuse some of the strategies for exploring data in Pandas we have learned about earlier.

Let's look into the questions when their seats are up again for the senators. Create a new dataframe with copy() from senators that only contains the 'Senator', 'State', 'Party', 'Occupation(s)', 'Years in Office' and 'Term up' columns. Call it senators_seatup.

Run `senators_seatup = senators[['Senator', 'State', 'Party', 'Occupation(s)', 'Years in Office', 'Term up']].copy()`.

senators_seatup = senators[['Senator', 'State', 'Party', 'Occupation(s)', 'Years in Office', 'Term up']].copy()       

Take a look at the first couple of rows of the data, and you will only find those columns you selected.

senators_seatup.head()

What are the types? Do you need to change them?

senators_seatup.dtypes

In my case, they were ok. They are only strings and ints - all in the right place.

Next, we would like to determine the next time an election is held. This information is in the 'Term up' column and there logically the smallest value. Assign that value to a variable next_election and pring it out by running 
```
next_election = senators_seatup['Term up'].min()
next_election
```

next_election = senators_seatup['Term up'].min()
next_election

Now, we select the rows/observations that are relevant for the next election and filter the senators_seatup rows with next_election. Assign the results to senators_seatup_next. Do you remember how to do this? If not check https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html for a quick reference.

senators_seatup_next = senators_seatup[senators_seatup['Term up'] == next_election]

Display all the senators whose seats are up.

senators_seatup_next

So far so good. Let's next group observations together to gain composite insights. Let's look at the senators per US state. Use senators_seatup_next and the columns 'State' and 'Term up' to display the number of terms that are up in the next election. Run `senators_seatup_next[['State', 'Term up']].groupby(['State'], as_index=False).count()`.

senators_seatup_next[['State', 'Term up']].groupby(['State'], as_index=False).count()

At least in 2021, there were quite a few senators up for re-election. How does it look for you? 

Finally, we wanted to look into the election challenges per party. Select 'Party' and 'Term up' and group by party to display the results with count(), please. You can do it ...

senators_seatup_next[['Party', 'Term up']].groupby(['Party'], as_index=False).count()

Republicans had far more seats to lose in 2021. You might see different results depending on the election you look at. Let's try and find out a little bit more about the senators up for re-election.  What is their median time in office if you group them by party? 

The agg function is the last very powerful Pandas tool we introduce: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html. Used together with groupby on party we can apply easily several functions like median and count in our case: `senators_seatup_next[['Party', 'Years in Office']].groupby(['Party']).agg(['median', 'count'])`. You should read this expression as a pipleine, where you first select two columns from senators_seatup_next, then group by one of the columns and aggregate the values with two functions.

senators_seatup_next[['Party', 'Years in Office']].groupby(['Party']).agg(['median', 'count'])

In 2021, the Democrats had much younger senators who had also served much shorter, which might indicate that they had much less time to secure the seat for their party. Your results will of course depend on the year you are looking at but can you find similar patterns? 

That's it for today's analysis of social communities with the additional bonus of learning a little bit about how to harvest data from the web, which is already advanced stuff. Thank you ...