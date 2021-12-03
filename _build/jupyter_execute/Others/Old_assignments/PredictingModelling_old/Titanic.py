# Predicting the survival of Titanic passengers

Taken from Datacamp Tutorial
https://goo.gl/53waiX 

When the Titanic sank, 1502 of the 2224 passengers and crew got killed. One of the main reasons for this high level of casualties was the lack of lifeboats on this supposedly unsinkable ship.Those that have seen the movie know that some individuals were more likely to survive the sinking (lucky Rose) than others (poor Jack). 

In the exercise, you wil apply machine learning techniques to predict a passenger's chance of surviving using Python.

## Kaggle

This example is famous, because Kaggle ran a competition with it. In 2010, Kaggle (https://www.kaggle.com/) was founded as a platform for predictive modelling and analytics competitions on which companies and researchers post their data and statisticians and data miners from all over the world compete to produce the best models. This crowdsourcing approach relies on the fact that there are countless strategies that can be applied to any predictive modelling task and it is impossible to know at the outset which technique or analyst will be most effective. Kaggle also hosts recruiting competitions in which data scientists compete for a chance to interview at leading data science companies like Facebook, Winton Capital, and Walmart. (https://en.wikipedia.org/wiki/Kaggle)

As data scienc Blogger Trevor Stephens writes: "So you’re excited to get into prediction and like the look of Kaggle’s excellent getting started competition, Titanic: Machine Learning from Disaster? Great! It’s a wonderful entry-point to machine learning with a manageably small but very interesting dataset with easily understood variables. In this competition, you must predict the fate of the passengers aboard the RMS Titanic, which famously sank in the Atlantic ocean during its maiden voyage from the UK to New York City after colliding with an iceberg." (http://trevorstephens.com/kaggle-titanic-tutorial/getting-started-with-r/)

Following the Kaggle competition, we need to import the training data first.

import pandas as pd

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

train.head()

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

test.head()

train.describe()

test.describe()

## Data exploration

Passengers that survived vs passengers that passed away:

from collections import Counter

print(Counter(train['Survived']))

Males and females that survived vs males and females that passed away:

males = train.loc[train['Sex'] == 'male']
print('Males:',Counter(males['Survived']))
      
females = train.loc[train['Sex'] == 'female']
print('Females:',Counter(females['Survived']))

## Data preparation/cleaning

Next we create the additional column child, and indicate whether child or no child

We have to do the same for sex, let's say maile is 0 and female is 1

And to Embarked, S is 0 and C is 1


import numpy as np

train['Child'] = np.where(train['Age'] <18.0, 1, 0)
train['Sex'] = np.where(train['Sex'] == 'female', 1, 0)
train['Embarked'] = np.where(train['Embarked'] == 'C', 1, 0)


## Modelling

We will use a decision tree to find features that help with the survival of Titanic passengers.

Decision trees build classification models in the form of a tree structure. They break down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. (http://www.saedsayad.com/decision_tree.htm).

So, we will automatically build something like this:

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('decision-tree.png')
imgplot = plt.imshow(img)
plt.show()

You need to load the scikit library to make decision trees.

from sklearn import tree

cols = ['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df = train[cols]
# remove Nans
df = df.dropna()
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

x = df[features]
# remove Nans
Y = df['Survived']

clf = tree.DecisionTreeClassifier()
clf.fit(x, Y)

Visualize the tree

fig = plt.figure(figsize=(100,50))

class_names = ['dead', 'survived']
tree.plot_tree(clf, 
                   feature_names= features,  
                   class_names=class_names,
                   filled=True)
plt.show()

## Predicting

1. Make a prediction using titanic_tree and the test data.
2. For Kaggle: Create a data frame with two columns: PassengerId and Survived. Survived contains your predictions.
3. Check the rows of the data frame.

Please, report at the end of the class on the Kaggle process. Can you find more competitions that interest you?

To make a prediction on the test set we first have to clean it as we did with the train set

test['Child'] = np.where(test['Age'] <18.0, 1, 0)
test['Sex'] = np.where(test['Sex'] == 'female', 1, 0)
test['Embarked'] = np.where(test['Embarked'] == 'C', 1, 0)

cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df2 = test[cols]
# remove Nans
df2 = df2.dropna()

x_test = df2[features]

y_pred = clf.predict(x_test)

We now calculate how many were predicted to survive

Counter(y_pred)