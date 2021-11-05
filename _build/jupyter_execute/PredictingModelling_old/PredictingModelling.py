# Predicting Modelling

Today, we will learn that machine learning is much less scary than science fiction will want to us to believe. This is not because we have benevolent machines, which only want our best, but simply because these machines are quite far away from living their own life without our input, as Skynet manages in 'Terminator' or the Machine in 'Person of Interest'. For the time being, machines still learn best when provided with human input. Furthermore, machines learn in most applications not because they want to start to understand the meaning of life and find out that humans are obstacles to true life, but because they learn to complete a particular task. Machines learn to be part of the workbenches of digital productions.

It is maybe less a link to artificial intelligence in science fiction than the fact that machines learn from our examples and need to be fed with large amounts of data to learn that makes machine learning an ethically difficult endeavour. Machine learning demands ever more data. Most aspects of our lives are recorded in vast data stores that are easily accessible to machines. Governments, businesses and individuals are recording and reporting all manners of information from the monumental to the mundane. As long as the activities can be transformed into digital formats, you can be certain that somebody will record it. 

In such a world, machines learn by consuming data and humans continuously add new digital methods of machine learning that can exploit this data. These can be some of the statistical methods we have already met or more advanced ones, we will meet today. The digital methods we learn about today have in common that they aim to predict new observations from old observations. They are all empirical and predictive using models.

Machine learning algorithms are all around you. They have tried to predict the outcomes of elections and referenda, can identify spam messages, predict crime and natural disasters, target donors and voters as well as finally have learned how to drive cars. Recently, they got it wrong quite often: http://www.kdnuggets.com/2016/11/trump-shows-limits-prediction.html

Many stories are told about the uses and abuses of machine learning. You can find some in the readings. Given how much machine learning is now part of our everyday life, it is maybe surprising that there are not even more stories. 

We also still lack an ethics of machine learning, which is developing so fast that it is difficult for laws and norms to stay up to date. There is, for instance, an on-going debate how biased machine learning algorithms are with regard to race and gender. Machine learning also has made it possible to identify people based on the region they live, the products they buy, etc. 

As a machine learning practitioner, you are often required to exclude revealing data that is ethically problematic, but this is not an easy task, as sometimes the connections are not obvious and might only be revealed after you have trained the machine to learn. 

## Background 1: The Data Science Process

### Social and cultural analytics and its data

Just like humans, machines use data to generalize. They abstract the data and develop its underlying principles, because humans tell them how. In the words of machine learning, machines form a model, which assigns meaning and represents knowledge. The model summarizes the data and makes explicit patterns among data. 

There are many different types of models. We have already seen some and others you will know from school. Models can be (statistical) equations, figures like graphs or trees, rules or clusters. Machines don't choose the type of models, we choose them for them when analysing the task at hand and the available data. 

The computer learns to fit the model to the data in a process called training. However, computational modelling does not end here. We also need to test the model in a separate testing process. The model thus does not include anything else but what can be found in the data already. It can nevertheless be interesting, as the model might surface connections that we did not recognize before. Newton discovered gravity this way by fitting a series of equations (a model) to observations of falling apples – if the myth is to be believed. Gravity was always there but it was observed for the first time in a model. 

Modelling is far from perfect. It generally involves some kind of bias or systematic error. Newton's laws of gravity are not as universal as he thought they would be. 

Errors like this do not have to be a bad thing, because they can lead the computer to be able to learn a better model, correcting previous mistakes. But generally, bias is to be avoided. Your reading includes the example where a machine learning algorithms learned to discriminate wolves and huskies from a series of online pictures. It achieved excellent performance until somebody found out that the decision was often based on whether snow can be found in the pictures’ background.

All learning has weaknesses and is biased in a particular way. Researchers are still looking for the universal model that is better than the rest of them but will probably never find it. Therefore, it is really important to understand how a model can overcome bias. This is the purpose of testing it on new data.

Unfortunately, especially in our domain of social and cultural analytics, models often fall short of desirable performance. Humans are difficult for computers and the data they produce and can be judged by is very noisy. This means that social and cultural data includes many errors because observations have not been measured correctly or maybe they are simply impossible to measure. How do you quantify, for instance, love? It seems impossible, but online match making agencies still make a business out of predicting love. 

Humans are also inconsistent and report data wrongly. Finally, especially in history we simply do not have data for all time periods or if we have data, it will include many missing values or will be badly captured according to diverse and sometimes contradictory standards. Often, the records have simply been lost. 

A final complication with data in social and cultural analytics that has only recently emerged is the limited access we have to the data. Because it is so valuable, it is kept behind the walls of company servers and is not shared.

So, machine learning is not artificial intelligence yet but a laborious collaboration between humans and machines that involves trying models and fighting with (bad) data. Otherwise, machine learning is a process that consists of a series of repeatable steps, which we will learn about today. In today's reading Schutt and O'Neil (2013), have given us an excellent overview of the art of data science.


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('process.png')
imgplot = plt.imshow(img)
plt.show()

According to the Figure, we first need to collect (raw) data in a form that we can process it. The next step explores the data and cleans it. People in data science like to emphasize that this is about 80% of the whole work. Then, we need a question we would like to answer with the data. This question will of course be at the beginning of our work but will likely also change after the initial exploration. Based on the question and the exploration, we start with the model and train it using a subset of the data. After training, we need to evaluate the model's performance by running a series of test predictions against test data. The result of the evaluation will then be used to improve the model's performance iteratively until we are satisfied that the model performs as best as possible, and decisions can be confidently made.

Before we experience the art of machine learning and prediction, let's quickly remind ourselves of what data is in the eye of the machine. Data generally describes a series of observations, which in R are generally captured in the rows of a data frame. Each observation is defined by its features (characteristics), which are the columns of a data frame. If a feature represents a characteristic measured in numbers, it is unsurprisingly called numeric. For instance, the years of the State of the Union Speeches were numerical. Alternatively, if a feature measures an attribute that is represented by a set of categories, the feature is called categorical or nominal. For instance, the colour codes red, green and blue are categorical. A special case of categorical variables is called ordinal, which designates a nominal variable with categories falling in an ordered list. Moview reviews on Netflix are, for instance, ordinal, because they only cover numbers from 1 to 5. 

You might also remember that we distinguished earlier supervised learning from unsupervised learning. We learned that clustering algorithms are an example of unsupervised learning where a machine discovers patterns/clusters in the data by itself. Today, we mainly work on the much larger group of supervised learning algorithms, where an algorithm is given a set of training data and then learns a combination of features that predicts certain behaviour such as whether an earthquake will take place soon or a crime will be committed. What we are trying to predict is also called a target variable. 

## Predicting Taste

Today, we will predict something that seems to define a human as inherently subjective. We will predict taste and in particular we will try to predict whether wine tastes good or bad. In the language of machine learning, this is a classification task. Our classification will predict whether any wine will fall into either one of two classes: good or bad wine. 

We will thus solve an ancient problem of philosophy, which interogates the subjectivity of taste or the aesthetic judgement (http://plato.stanford.edu/entries/aesthetic-judgment/). For the German philosopher Kant, taste judgments are universal and subjective at the same time. A key part of his Critique of Judgement, Kant demands more from taste than we are generally willing to attribute to it: 'Many things may for [a person] possess charm and agreeableness — no one cares about that; but when he puts a thing on a pedestal and calls it beautiful, he demands the same delight from others. He judges not merely for himself, but for all men, and then speaks of beauty as if it were a property of things. (…). He blames them if they judge differently, and denies them taste, which he still requires of them as something they ought to have; (…).' (http://oll.libertyfund.org/titles/kant-the-critique-of-judgement, §7). 

Today, we will use the machine to find out how something can be subjective and universal at the same time.

To illustrate how machines classify, let’s first go through a simplified dataset that helps us understand taste. Because we like it sweet and crunchy, we create a training dataset by tasting 1,000 foods and record for each of them how crunchy and how sweet they were. Both crunchy and sweet are ordinal features with a range from 1 to 10. Next, we would like to map this data into a so-called feature-space with 2-axes: one for crunchiness and one for sweetness. 

This example is taken from the excellent Lantz (2013) (Machine learning with R. Packt Publishing Ltd.), which could be a good reference for you to continue working through machine learning using R. Be warned, however, it is fairly advanced. But you will get there! 

Lantz produced a nice visualisation of such a feature space with a few example foods:

img = mpimg.imread('lantz-1.png')
imgplot = plt.imshow(img)
plt.show()

Lantz notices that in this feature space 'similar types of food tend to be grouped closely together. (…), vegetables tend to be crunchy but not sweet, fruits tend to be sweet and either crunchy or not crunchy, while proteins tend to be neither crunchy nor sweet.' (p. 68). Similarity is thus based on the distance of the items in the feature space.


img = mpimg.imread('lantz-2.png')
imgplot = plt.imshow(img)
plt.show()

Next, we taste for the first time a tomato and add it to the feature space.

img = mpimg.imread('lantz-3.png')
imgplot = plt.imshow(img)
plt.show()

Based on this mapping how would we classify the tomato? Is it a vegetable or a fruit? The figure is not very conclusive because we cannot really determine which group the tomato is closer to in the feature space.

You have just learned how a machine would learn and think about the tomato as well as which decisions it would have to make to understand tomatoes. Machines learn similarities in feature spaces using distances.

## Identify the problem: Machine-tasting  Wines

Let's go next through our example of tasting wines next and explore the individual steps of machine learning more closely. The data comes from http://archive.ics.uci.edu/ml/. Check it out. It’s a famous repository for machine learning datasets. The wine data (http://archive.ics.uci.edu/ml/datasets/Wine+Quality) consists of 2 CSV files, one for white wines and another for red ones.

The two datasets are related to red and white variants of the Portuguese Vinho Verde wine, and were first used in Cortez et al (2009) (Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4): 547-553).

I first thought of this example, when I learned that somebody else had already 'outsmarted' the best wine experts with machine learning: https://www.datanami.com/2015/02/20/outsmarting-wine-snobs-with-machine-learning/. Today we try and reproduce his approach.

The first step for us is to download the data so that we can work with it. We have done this before but in the previous tutorials, I have mostly loaded the data for you. So, it’s best to repeat the steps again in detail.
Perhaps the most common data format of freely available data are Comma-Separated Values (CSV) files, which, as the name suggests, uses the comma as a delimiter. CSV files can be imported to and exported from many common data repositories. To load CSV into python, we use pandas read_csv() function. You use it by specifying a path to the file you want to import, e.g. /path/to/mydata.csv, when calling the pd.read_csv() function after importing pandas again. Here we use it to load the data directly from the web.

### Collect Data

import pandas as pd

red = pd.read_csv(
     'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', delimiter=';')
white = pd.read_csv(
     'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', delimiter=';')

This creates two data frames, one for each type of wine. read_csv() directly accesses the data frames from the web, as you can see, because it uses an http address. Please, note that I generally would advise you to download the data first, as you can never be certain whether you will always have a working Internet connection.
As we would like to follow the work by Cortez et al. as closely as possible, we next add another feature/column to capture the colour of the wine.

for row in red:
    red['color'] = 'red'

for row in white:
    white['color'] = 'white'

Now we create single data frame for all the wines and declare that colour is factor. pd.concat is a function to bind two data frames row by row.

red.index = range(4899, 6498)

wines_df = pd.concat([white, red])

wines_df

This completes our first step, the data acquisition/collection. It is fairly easy, as we reuse existing material. The data is also complete, and we do not have to take care of any missing values.

As described earlier, we want to the machine to learn how to taste good and bad wine. Let’s take a first look at the dataset using .info() and head()

wines_df.info()

wines_df.head()

There is a column called quality, which matches our classification task. We will use this column as the classification 'target'. Quality is an ordinal feature from 1 to 9 with 9 indicating top quality. Now, let's see how quality values are distributed. We could simply run table to get the frequencies for each quality class, but we decide to plot the classes using plot.hist().

wines_df['quality'].plot.hist()

In order to make our life a little easier, we now would like to reduce the 9 quality classes to 2 (good or bad). This is also part of the original example. 

import numpy as np

wines_df['quality'] = np.where(wines_df['quality'] < 6, 'bad', 'good')

We have overwritten the original quality column with a new quality factor with 2 levels. Let’s see how this is distributed with value_counts()

wines_df['quality'].value_counts()

Unfortunately, we now have many more ‘good’ quality wine observations, which might be a problem later when we start training a model. Why do you think this might be the case? But it can’t be helped and we go on analysing.

It's time to prepare our data for its machine learning adventures.

### Prepare Data

The next step is very important for many machine learning algorithms based on feature spaces. We need to standardize the features, as the distances in the space are dependent on how the features are measured. In particular, if certain features have much larger values than others, the distance measurements will be strongly dominated by the larger values. This wasn't a problem for us before with the food data, as both sweetness and crunchiness were measured on a scale from 1 to 10. But suppose we added another measure on a scale from 1 to 1,000,000. This measure would dwarf the contribution of the other scales. The distances in the feature space would get out of scale.

We only need to normalize numeric data. Looking back at the results from str(wines_df), columns/features 1 to 11 are numeric. Next we, define a function to normalise these so that they are all on a scale between 0 and 1. We use the so-called min-max normalisation. Consider an example, where the residual sugar of wine is say 50, while we want to transform this to the range 0 to 1. So first we find the maximum value of residual sugar which is in our example 100 and the minimum value of residual sugar, say 20, then the new scaled value for will be: (50-20)/(100-20)=0.375. Can you see why this value is guaranteed to be between 0 and 1?

Let’s define a function that takes care of the normalization for us. You hopefully remember how you can define your functions in Python?

def normalize(x, li):
    n = ((x - min(li)) / (max(li) - min(li)))
    return n

x = 50
li = [20,50,100]
normalize(x, li)


Now, we apply normalize to all the numeric columns wines_df[1:11]. 

wines_df[0:10]

# get all numeric columns
newdf = wines_df.select_dtypes(include='number')

# create new dataframe for normalized numbers
ndf = wines_df.select_dtypes(include='number')

# quite a complicated function? 

for col in range(11):
    li = newdf.iloc[:,col]
    for i, x in enumerate(li):
        n = normalize(x,li)
        ndf.iloc[i, col] = n

ndf.head()

Finally, let’s add the quality column to the new normalized data frame. This time it only contains good and bad.

ndf['quality'] = wines_df['quality'].values


We are now satisfied with the data, done our cleaning and all preparations. We can start the modelling process in order to predict how a wine will taste. First, we need to check how many wine quality observations we have in total.

### Define the Training Data

ndf

We have 6497 wine quality observations with 2384 labelled bad and 4113 labelled good.
Because we aim to predict new things, our next step should be to find out about things we do not already know and how the model would be able to predict unknown data. If we had access to more wine data, we could apply our model to unknown wine observations and see how well the predictions compare to new wines. But we cannot know about data we do not have. So, we simulate such a scenario by dividing our data into a training dataset that will be used to build the model and a test dataset (as described above). We will use the test dataset to simulate the prediction and find out how well our model behaves.


We will use 75% of our data for the training and 25% for testing. First we randomly mix the data with  to ensure that all qualities are evenly distributed in both training and test data. Then, we use the first 4549 records (~75%) for the training dataset and then rest for the test data. Remember that data is extracted from data frames using the [row, column] syntax. A blank value for the row or column value indicates that all rows or columns should be included.

shuf = ndf.sample(frac=1)
train_n = int(0.75 * len(shuf))
test_n = int(0.25 * len(shuf))

train_set = shuf[:train_n]
test_set = shuf[train_n + 1:]

We are ready to model and because things are looking good with go directly to one the most advanced machine learning technique that uses the human brain as an inspiration.


## Modelling and Predicting 

### Background 2: Neural Networks

With the training data, we are ready to start learning a model for tasting wine. To classify our test instance, we will work with the best that current machine learning has to offer. We employ the help of neural networks, machines assembled in similar ways as the hundreds, thousands or millions of brain cells. These machines are supposed to learn and behave in a similar way to human brains. Kant would be proud of us – maybe. 

img = mpimg.imread('neural-networks-1.png')
imgplot = plt.imshow(img)
plt.show()

Each neuron is made up of a cell body with a number of connections coming off it. These are numerous dendrites (carrying information toward the cell body) and a single axon (carrying information away). But computers are not alive. They are mechanical boxes and made not of the complex chains of brain cells, which are densely interconnected in complex and parallel ways - each one connected to perhaps 10,000 other brain cells. Computers are designed to store lots of data and rearrange that – as we have done many times and need instructions for that. To the day, we do not fully understand how brains learn. They can spontaneously put information together in astounding new ways and forge new connections. No computer currently comes close to that.

The basic idea behind a neural network is to simulate those densely interconnected brain cells inside a computer so you can get it to learn things, recognize patterns and make decisions. Neural networks learn to improve their own analysis of the data. But neural networks remain mathematical equations and mean nothing to the computers themselves – unlike our own brain activities. They are still just highly interconnected numbers in boxes who constantly change. 

A typical neural network has anything from a few dozen to hundreds, thousands, or even millions of artificial neurons called units arranged in a series of layers, each of which connects to the layers on either side. Some of them are input units. In our case, these will be defined by the data for each feature in each observation. Each feature forms one input unit. Neural networks also have an output layer that responds to the information that is learned. In our case, these are the quality judgments we make with regard to the wines. 

img = mpimg.imread('neural-networks-2.png')
imgplot = plt.imshow(img)
plt.show()

In-between the input units and output units are one or more layers of hidden units, which, together, form the majority of the artificial brain. The connections between one unit and another are represented by a number called a weight, which can be either positive (if one unit excites another) or negative (if one unit suppresses or inhibits another). The higher the weight, the more influence one unit has on another. Inputs are fed in from the left, activate the hidden units in the middle and feed out outputs from the right.

But information flows backwards from the output units, too. For a neural network to learn, there has to be an element of feedback involved – just like we humans learn. With feedback, we compare what we tried to achieve with what we actually achieved and adjust our behaviour accordingly. Neural networks learn things in exactly the same way with a feedback process called backpropagation. Because we know from the training data the output we tried to achieve, we can compare it with the calculated values and modify the connections in the network to improve the outcome, working from the output units through the hidden units to the input units. Over time, this backpropagation causes the network to learn until a stable state is achieved. In our case, the network will lean how we taste wine.

Neural Networks have become synonymous with the recent success of artificial intelligence. So much so that the Guardian declared that 2016 was the year of AI because of advancements in Neural Networks (https://www.theguardian.com/technology/2016/dec/28/2016-the-year-ai-came-of-age).

### Modelling and Predicting

Fortunately for us, we do not have to implement Neural Networks by ourselves but can rely on many existing algorithms in Python. 

One of the most common libraries in python used for machine learning is scikit-learn: https://scikit-learn.org/stable/

To instill scikit learn, type pip3 install scikit-learn into your terminal.

For now we will use scikit's MLPclassifier, a Multi-layer Perceptron classifier.


import sklearn as sk
from sklearn.neural_network import MLPClassifier

We first divide our train and test set into predictor and target variables. 

X_train = train_set.iloc[:,:-1].values
Y_train = train_set.iloc[:,-1:].values

X_test = test_set.iloc[:,:-1].values
Y_test = test_set.iloc[:,-1:].values

And then build the classifier

classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)

classifier.fit(X_train, Y_train)

### Predicting 

Next, we can start predicting unknown behaviour, which - as said - we simulate with the test dataset. 

y_pred = classifier.predict(X_test)

Let's check out the details of our predictions and compare predictions with test data using scikit's confusion matrix function which prints the true positives, false negatives, false positives an true negatives consequently.

## Evaluate Model

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, Y_test)

print(cm)

# tp, fn, fp, tn

Next, let's look at the accuracy of our prediction.

Accuracy is defined as the number of times our predictions have been correct compared to the overall number of predictions. So, we take all cases where the predictions where right in the above table (bad-bad and good good) and compare these with the overall number of observations in the test data or len(test_set). Please replace in the calculation below the numbers you have got.

(432 + 835) / len(test_set)

~75% of our predictions are correct. Please, note that the exact number can be either a bit higher or lower depending on the random test and training datasets. 

Not bad – especially considering that most wine experts would probably not be able to agree to such a degree. However, we would of course like to improve on our predictions. So, let's investigate this further. 

From the table, we can see that the machine is much better at predicting good quality wine rather than bad one. This should not be surprising, since we already know that we do not have enough training data for bad wine. Let's plot our results next.

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(classifier, X_test, Y_test) 

### Interpret the Results

In order to further interpret the model, a good approach is to understand which features have influenced the models behaviour and which features are redundant because the results they support are supported by other features. This way we get closer to the secret why people like certain wines. Let's find out first which features influence the quality decisions most. 

from sklearn.inspection import permutation_importance

r = permutation_importance(classifier, X_test, Y_test,
                          n_repeats=30,
          random_state=0)

for i in r.importances_mean.argsort()[::-1]:
     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{ndf.columns.values[i]:} "
              f"{r.importances_mean[i]:.3f}")


imps = r.importances_mean

features = pd.DataFrame(imps, ndf.columns.values[:11], columns = ['importance'])
features

features.sort_values(by=['importance']).plot.bar()

No wine feature really dominates human taste, but density is the most important one – followed by residual sugar and total sulfur dioxide. 


In order to check for redundant features that we do not need for the prediction, we can use correlation for numerical features. Remember that a correlation indicates the extent to which two or more features fluctuate together. A positive correlation indicates the extent to which those variables increase or decrease in parallel. The higher the correlation between variables therefore the easier it will be to use just one of them, as the others do not influence the overall outcome. 

We will plot a heatmap of the correlations using seaborn,

The correlation coefficient has values between -1 to 1

— A value closer to 0 implies weaker correlation (exact 0 implying no correlation)

— A value closer to 1 implies stronger positive correlation

— A value closer to -1 implies stronger negative correlation


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))
cor = train_set.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

corr() delivers the correlations between all features in train_set and sns.heatmap plots them. In the plot, you can clearly see that the two sulfur.dioxide measures are correlated. A negative correlation indicates the extent to which one variable increases as the other decreases. According to the plot density and alcohol drive taste opinions apart. 

The next step would be to try and improve the model performance. We could, for instance, make the network more complex or change the normalisation. The possibilities are literally endless. This kind of work is what keeps an analyst really occupied. In our case this might be difficult though as we do not have enough data on bad wines. We could try and get more data and organise another tasting competition, but going to Portugal is expensive. We rather look at a recent innovation of the neural network called 'deep learning' next. Deep learning is essentially a way to learn much more complex neural network architectures, more layers of hidden neurons and more complex connections. 

There is an excellent TED talk on deep learning: https://www.youtube.com/watch?v=xx310zM3tLs Google, Facebook, Bing and all the others currently invest millions into new services based on deep learning. Facebook, for instance, has released DeepText (http://www.wired.co.uk/article/inside-deeptext-facebook-deep-learning-algorithm) to understand the textual content of millions of posts. 

## Predicting and Modelling using Deep Learning

To create deep learning models in Python we will use Keras, a deep learning API in extension to scikit-learn

Before we do this, we have to change the class (quality) to binary

nndf = ndf.copy()

nndf['quality'] = np.where(nndf['quality'] == 'bad', 0, 1)

nndf

We import the necessary modules from scikit and keras

I you haven't downloaded keras yet, download it by typing pip3 install keras in your terminak

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical

Like before, we get the x and y values and split the dataset into a train and test set, this time using scikit's train_test_split function.

X = nndf.iloc[:,:-1].values
y = nndf.iloc[:,-1:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
print(X_train.shape); print(X_test.shape)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(500, activation='relu', input_dim=11))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
            metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20)

pred_train= model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   

pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1])) 

## Decision Boundaries

As a final experiment, let's try and generate those decision boundaries you read about in your readings, which decide in our case about the wines and whether they are good or bad. 

We would like to create a feature space to visualise this decision. To this end, we first need to map our 11 features into 2 features, because we cannot visualise an 11-dimensional space that humans could read. A standard strategy for reducing the dimensions of a problem like this is to apply Principal Component Analysis (PCA) (https://en.wikipedia.org/wiki/Principal_component_analysis). It is a fairly complicated method. If you are interested, we can discuss it in class and there are also good explanations in your readings. Here, it is enough to know that for each wine observation it will find its position in a lower-dimensional feature space. 


# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

nnndf = nndf.copy()
# scale data
x = nnndf.iloc[:,:-1].values
y = nnndf.iloc[:,-1:].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

df = pd.concat([principalDf, nnndf[['quality']]], axis = 1)
df

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)

targets = [1,0]
colors = ['g', 'r']

for target, color in zip(targets,colors):
    indicesToKeep = df['quality'] == target
    ax.scatter(df.loc[indicesToKeep, 'principal component 1']
               , df.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
    
qualities = ['good', 'bad']
ax.legend(qualities)
ax.grid()

Next we will visualise the decision boundaries. 

from sklearn.svm import SVC
from sklearn import svm

# How to plot the decision boundaries?
# this is from https://stackoverflow.com/questions/43778380/how-to-draw-decision-boundary-in-svm-sklearn-data-in-python

model = svm.SVC(gamma=0.001,C=100.0)
rng = np.random.RandomState(0)
s = model.fit(principalComponents, y)

# generate grid for plotting
h = 0.2
x_min, x_max = principalComponents[:,0].min() - 1, principalComponents[:, 0].max() + 1
y_min, y_max = principalComponents[:,1].min() - 1, principalComponents[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))

# create decision boundary plot
Z = s.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx,yy,Z,cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(principalComponents[:,0],principalComponents[:,1],c=y)
plt.show()