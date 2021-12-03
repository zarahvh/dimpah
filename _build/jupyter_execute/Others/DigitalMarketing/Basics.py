# Digital marketing and Python - The basics

## Digital Marketing and Data Science

Python can easily perform complex data manipulations, such as regression analysis, market basket research, prediction, cluster analysis, customer segmentation and much more, which are common in marketing analytics. Using specific libraries, you can access your data from Google Analytics, Web Logs, and many more marketing-related software, helping bring all your data into one place for more thorough analysis.

For a lot of marketing analysis, you will be dealing with average sized data sets and not necessarily the Big Data size that is usually associated with data science. Although data science skills aren’t necessary to be a digital marketer, they will help you analyse and better understand the data so that you can make data-driven business decisions that will improve your marketing methods and therefore improve performance. They will help you rely less on tools. 

To find out more about data science in marketing check out:

- https://www.ama.org/publications/MarketingNews/Pages/data-science-latest-in-demand-skill-marketing.aspx
- https://lab.getapp.com/marketing-analytics-data-analysis-in-marketing/
- DATACAMP's Marketing classes. Most of these notebooks are based on these tutorials

## Standard Data Exploration

The purpose of this first notebook is to introduce you to a typical marketing dataset and demonstrate that the techniques you already know apply here as well. 

We load example sales data.

import matplotlib.pyplot as plt
import pandas as pd

salesdata = pd.read_csv("salesData.csv")
salesdata.describe()

salesData is about customers and their sales in the last three months. Otherwise:

- id: identification number of customer
- mostFreqStore: store person bought mostly from
- mostFreqCat: category person purchased mostly
- nCats: number of different categories
- preferredBrand: brand person purchased mostly
- nBrands: number of different brands

Next we want to visualise the correlation in the dataset. Do you remember .corr() ?

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))
cor = salesdata.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

The following boxplots show how salesThisMon is dependent on the preferredBrand and mostFreqStore.

plt.figure(figsize=(12,10))
sns.boxplot(x=salesdata['mostFreqStore'], y=salesdata['salesThisMon'])

plt.figure(figsize=(12,10))
sns.boxplot(x=salesdata['preferredBrand'], y=salesdata['salesThisMon'])

Which features are  well suited to explain the sales of this month?

There seems to be definitely something about the Stockton store, while for the brands the situation seems to be much less obvious.

As a final exercise, let's check out a quick linear regression analysis. We would like to investigate further the relationships of days since last purchase.

sns.lmplot(data=salesdata, x='daysSinceLastPurch' , y='salesThisMon')

There is obviously a negative relationship between the days since the last purchase and the sales this months. 

Let's control for brand next.

sns.lmplot(data=salesdata, x='daysSinceLastPurch' , y='salesThisMon', hue='preferredBrand')

As you can see, for some brands, the time delay effect is less strong.
    

I hope you can  see how the methods we have used in the module can easily be applied to digital marketing data. Next we will investigate several examples of specific digital marketing analysis.