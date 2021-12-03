# Choice Modelling

We make choices every day, and often these choices are made among a finite number of potential alternatives. For example, do we take the car or ride a bike to get to work? Will we have dinner at home or eat out, and if we eat out, where do we go? Scientists, marketing analysts, or political consultants, to name a few, wish to find out why people choose what they choose. We can model these choices as we can model other things.

Choice modelling is about following the decision process of individuals and understanding the kinds of decisions they make: https://en.wikipedia.org/wiki/Choice_modelling/ and https://www.r-bloggers.com/choice-modeling-with-features-defined-by-consumers-and-not-researchers/. 

Choice modelling is often closely linked to surveys. For instance, research into chocolate preferences.

## Use Case: Chocolates

In this survey, a respondent chooses between three chocolate bars with different brand, price and type. 

The data shows one choice in each row. 
The data is in wide format, and each row represents one choice. 
In wide format, categorical data is always grouped. It is easier to read and interpret as compared to long format, which we will introduce later. Wide format is often the output if you use tools like surveymonkey, etc.

import matplotlib.pyplot as plt
import pandas as pd

chocolate_wide = pd.read_csv("chocolate_choice_wide.csv")
chocolate_wide.describe()

Check out some details. How many choices are part of the chocolate_wide data frame?

chocolate_wide.head()

You can also select individuals with subset. In this case, we are looking at subject 2408 in trial 3.

subject = chocolate_wide.loc[(chocolate_wide['Subject'] == 2408) & (chocolate_wide['Trial'] == 3)]
subject

### Converting from wide  to long data

Our algorithms need the long data. In the long vertical format, every row represents an observation belonging to a particular category. You can use pandas wide_to_long()

With reshape directions indicate the direction of the data transformation (in this case long), while stubnames indicates the columns you would like to summarize. Here, it is Brand, Price and Type. i gives the  columns names that you want to keep. j, finally, is the variable in long format that differentiates multiple records from the same group or individual.

chocolate_df = pd.wide_to_long(chocolate_wide, stubnames=['Brand','Price','Type'], i=['Subject', 'Trial', 'Selection', 'Response_Time'], j='Alt').reset_index()
chocolate_df

For survey data, we often want the individual/subject to come first, then the trial/experiment and finally the choice. Let's order the data so that this will be the case.

chocolate_df = chocolate_df[["Subject", "Trial", "Selection", "Response_Time", "Alt", "Brand", "Price", "Type"]]

chocolate_df

The variable Alt labels the chocolate alternatives, and the  variable Selection indicates the chosen alternative. Let's transform Selection into a logical variable that indicates whether a certain alternative was chosen. Do you know how?

chocolate_df.loc[chocolate_df['Selection'] != chocolate_df['Alt'],'Selection'] = False 
chocolate_df.loc[chocolate_df['Selection'] == chocolate_df['Alt'],'Selection'] = True 


chocolate_df

A few visualisations at the end.

TRUE = chocolate_df.loc[chocolate_df['Selection'] == True]
true_type = TRUE['Type'].value_counts().to_list()
FALSE = chocolate_df.loc[chocolate_df['Selection'] == False]
false_type = FALSE['Type'].value_counts().to_list()

df = pd.DataFrame([true_type, false_type], columns=[ 'White', 'Dark w/ Nuts', 'Milk w/ Nuts', 'Dark', 'Milk'])
df.plot.bar(stacked=True)

from statsmodels.graphics.mosaicplot import mosaic

brands = chocolate_df['Brand'].tolist()
selections = chocolate_df['Selection'].tolist()
data = pd.DataFrame({'brand': brands, 'selection': selections})
mosaic(data, ['brand', 'selection'])

### Conjoint Analysis

We have just done the beginning of a conjoint analysis. Conjoint analysis is a set of market research techniques that measures the value the market places on each feature of your product and predicts the value of any combination of features. Conjoint analysis is, at its essence, all about features and trade-offs. With conjoint analysis, you ask questions that force respondents to make trade-offs among features
- Determine the value they place on each feature based on the trade-offs they make
- Simulate how the market reacts to various feature trade-offs you are considering

https://www.pragmaticmarketing.com/resources/articles/conjoint-analysis-101
https://www.coursera.org/lecture/uva-darden-bcg-pricing-strategy-customer-value/conjoint-analysis-steps-1-3-pRWBU

## Modelling

Statsmodels

import statsmodels.api as sm

X = chocolate_df[['Brand', 'Type', 'Price']]
X = pd.get_dummies(data=X, drop_first=True)
Y = chocolate_df['Selection']
X = sm.add_constant(X)

model = sm.Logit(Y.astype(float), X.astype(float)).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)

The coef indicates the significance of the features. White Chocolate is, e.g, -1.7, which means people really don't like it.

The ratio of coefficients usually provides economically meaningful information. E.g., an interesting attribute is the Willingness to pay (WTP) or the maximum price at or below which a consumer will definitely buy one unit of a product (https://en.wikipedia.org/wiki/Willingness_to_pay).

We can calculate the wtp by dividing the coefficient vector by the negative of the price coefficient 

coefs = model.params
coefs/-(coefs[1])

A nice trick to get a better overview of prices is to factorize them and then model for price levels. Let's change the Price variable to a factor in the chocolate data.

# chocolate_df['Price'].factorize
chocolate_df['Price'] = chocolate_df['Price'].astype(object)
ch_df = chocolate_df[pd.notnull(chocolate_df['Price'])]
ch_df

# X = ch_df[['Brand', 'Type', 'Price']]
# X = pd.get_dummies(data=X, drop_first=True)
# Y = ch_df['Selection']
# # X = sm.add_constant(X)

# model = sm.Logit(Y, X).fit()
# predictions = model.predict(X) 

# print_model = model.summary()
# print(print_model)

# what does the factorizing do?
# Why does logistic regression not work?