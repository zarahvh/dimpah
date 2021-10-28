# Customer Relationship Management

## Survival Analysis

Survival analysis corresponds to approaches  to investigate the time it takes for an event of interest to occur. It's one of the oldest analytics fields going back to the 17th century. 

Survival analysis is used in a variety of field such as:

- Cancer studies for patients survival time analyses.
- Sociology for event-history analysis
- Digital marketing to answer questions such as after ordering for the first time in an online shop, when do customers place their second order?

### Use Case: Online Shop

Our example is data about customers in an online shop. dataNextOrder contains: boughtAgain with a value 0 for customers with only one order and 1 for customers with a second order already. If a person has ordered a second time, you will see the number of days between the first and second order in the variable daysSinceFirstPurch. For customers without a second order, daysSinceFirstPurch contains the time since their first order. The data also contains information about whether a voucher was used, the gender and whether the item was returned.

Let's check it out:

import matplotlib.pyplot as plt
import pandas as pd

dataNextOrder = pd.read_csv("dataNextOrder.csv")
dataNextOrder.head()

dataNextOrder['daysSinceFirstPurch'].plot.hist()

boughtagain = dataNextOrder.loc[dataNextOrder['boughtAgain'] == 1]
notboughtagain = dataNextOrder.loc[dataNextOrder['boughtAgain'] == 0]

boughtagain['daysSinceFirstPurch'].plot.hist(title='Bought again: Yes', color = 'red')
plt.show()
notboughtagain['daysSinceFirstPurch'].plot.hist(title='Bought again: No')
plt.show()

You can see that from about <50 days, the shop should be concerned whether a costumer returns/survives.

### Survival Models

In survival analysis, each observation has one of two states: either an event occured, or it didn't occur. But you don't know if it occurs tomorrow or in three years.

### Analysis 

We are going to apply the very basic technique called Kaplan-Meier Analysis. Kaplan-Meier analysis is used to analyze how a given population evolves with time. This technique is mostly applied to survival data and product quality data. 

The Kaplan-Meier method is a descriptive methods of survival analysis, and allows you to quickly obtain a population survival curve and essential statistics such as the median survival time. In our case 41 days.



from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter() 

kmf.fit(dataNextOrder['daysSinceFirstPurch'], dataNextOrder['boughtAgain'],label='Kaplan Meier Estimate')

## Create an estimate
kmf.plot(ci_show=False)

We can clearly see between 40-50 days the chance for a further purchase is halved. As you can see we haven't taken any of the other variables into consideration. we can also easily check whether, e.g., vouchers have an influence on the next purchase - as one might expect.

groups = dataNextOrder['voucher']   

novoucher = (groups == 0)      
voucher = (groups == 1)

kmf.fit(dataNextOrder['daysSinceFirstPurch'][novoucher], dataNextOrder['boughtAgain'][novoucher], label='No Voucher')
vplot = kmf.plot()

## fit the model for 2nd cohort
kmf.fit(dataNextOrder['daysSinceFirstPurch'][voucher], dataNextOrder['boughtAgain'][voucher], label='Voucher')
kmf.plot(ax=vplot)

Maybe surprisingly, customers using a voucher are taking longer fo second order. What are they waiting for?