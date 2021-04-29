
# Exploratory Data Analysis with Python

##  Introduction

This project trains you on the essentials of exploratory data analysis (EDA) with Python. There are many excellent resources and books. For a recent introducion, check out Suresh Kumar Mukhiya, Usman Ahmed, Hands-On Exploratory Data Analysis with Python, Pack Publishing, 2020.

Notebooks are particularly suited for EDA. Let's repeat quickly why.


## Notebooks

Notebooks are interactive environments to work and share your code for a project and reuse the recipes. They combine plain text with rich text elements such as graphics, calculations, etc. and executable code.

You are  working with a sort of text editor in which you indicate your code chunks and your pure text components. The document is self contained and fully reproducible which makes it very easy to share.

What's great about working with these notebooks is the fact that you can follow up on the execution of your code chunks, thanks to visual feeedbacks on the completion of code when you're executing large chunks or multiple chunks at once. 


### Code Execution

Notebooks have options to run a code chunk or run the next chunk, run all code chunks below and above. In addition to these options, you can also choose to restart the Python kernel and run all chunks or to restart and to clear the output. 

You have the option to run a single cell, to run several cells or to run all cells. You can also choose to clear the current or all outputs. The code environment is shared between code cells.

### Notebook Sharing

You can easily create an HTML file from the notebook that you can share with other people that do not have the same environment set up.

## Example project

EDA is sometimes treated as a step towards a deeper predictive analyis, but it really is a very powerful kind of analysis in its own right - especially if the dataset is not very large or very complex.


In the example project, we continues our debate today on analysing votes in parliaments with Python. We will use the dataset by Erik Voeten in 'Data and Analyses of Voting in the UN General Assembly', Routledge Handbook of International Organization, edited by Bob Reinalda (published May 27, 2013)
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2111149. 

We will analyse the voting behaviour of countries in the UN.

Let's start by loading the dataset. 

#Run the code below

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

#Run the code below

votes = pd.read_csv("../data/Votes.csv")

Check out the dataset with head().

votes.head()

Each row in the data is a country-vote pair. RCID is the row call for each vote. The session columns stands for the session, while 'vote' is the country's vote:

- 1 = Yes
- 2 = Abstain
- 3 = No
- 8 = Not present
- 9 = Not a member

Let's remove those observations we are not interested in, which are the votes 8 and 9: 'Not present' and 'Not a member'. 

Only keep those votes that are in 1, 2 or 3.

votes = votes[votes.vote.isin([1,2,3])]
votes.head()

The session columns start from 1946, when the UN was founded but now is simply the number of session. To make this clearer we add 1945 to the column values and create a new column year. Create a new column 'year' to do so.

votes['year'] = 1945 + votes.session

votes.head()

The country codes in the ccode column are what is called Correlates of War codes (ccode). But it would be better to have recognizable country names. You can use the countrycode package to translate. 

I have created a spreadsheet with the relevant informatio. Well, somebody else has and I just copied it. Run the next cell to create a mapping table.

#Run the code below

ccode_url = 'https://docs.google.com/spreadsheets/d/1DCA8DaKzUx4i-k6-QJTIW1DlvunvqJ2JKaIBa8DkJI4/export?gid=0&format=csv'
   
ccode_df = pd.read_csv(ccode_url) 

ccode_df.head()

Please use the information in ccode_df to create a column 'country' in votes with the country name.

votes = votes.merge(ccode_df[['ccode','statenme']], on=['ccode'])
votes = votes.rename(columns={"statenme": "country"})

votes

## Data exploration

Now, let's make this a little more interesting. We want to count the total number of yes and no votes per year as well as the proportional number of yes and no votes. 

Create a new dataframe by_year that contains the year, the vote column, the total number of votes per year and 'prop', a column containing the proportion of yes and no votes.

#  col 1 = year
#  col 2 = which vote (1,2 or 3)
# col 3 = total votes (for all)
#  col 4 = percentage of those votes = 'vote'


Now let's plot the data. Simply plot the number of yes votes per year. 

Do you see a pattern?



## Case Studies: Comparison of Yes Votes

Finally, let's try a larger case study and investigate the proportion of yes votes for a few countries. This is already a complete small recipe.

We first define the country list containing the ones we are interested in.

#Run the code below

countries_of_interest = ["Australia", "Brazil", "India"]

To change the countries you are interested in, just change the list above.

Now, we group not just by year but also by country. We would like to create the same output of yes and no votes, proportions and total again. 

Call the resulting dataframe by_year_country.

by_year_country = votes.groupby(['year','country', 'vote'])['ccode'].count().reset_index()
by_year_country = by_year_country.merge(votes.groupby(['year'])['ccode'].count().reset_index(), on = 'year')
by_year_country['prop'] = by_year_country.ccode_x / by_year_country.ccode_y
by_year_country = by_year_country.rename(columns={"ccode_y": "total"})
by_year_country.drop('ccode_x', axis=1, inplace=True)
by_year_country

# col 1 = year
# col 2 = which country
# col 3 = what did they vote
# col 4 = How many votes
# col 5 = what is the proportion of their vote for that 'vote', not the total

filtered_countries will be those that have voted yes at least once and are part of the countries we are interested in.

Create the dataframe.

# if vote = 1 and country is part of the interest countries

### Visualisation

Create a graph with the proportinal yes vote per country of interest per year.

Please, use the seaborn package.

#  sns scatterplot

That's it. Always remember how easy it is to change notebooks to suit your need.