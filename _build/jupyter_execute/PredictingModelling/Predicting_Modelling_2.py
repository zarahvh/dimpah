## Predicting and Modelling using Deep Learning

Deep learning is what moves people in AI at the moment. It is responsible for many breakthroughs. There is an excellent TED talk on deep learning: https://www.youtube.com/watch?v=xx310zM3tLs Google, Facebook, Bing and all the others currently invest millions into new services based on deep learning. Facebook, for instance, has released DeepText (http://www.wired.co.uk/article/inside-deeptext-facebook-deep-learning-algorithm) to understand the textual content of millions of posts. 

Deep learning is neural networks on steriods with many more neurons, hidden layers, connections, etc. These complex network structure could only recently be built. Even more recent are framework such as Keras in Python (https://keras.io/) that makes building deep learning models easier. 

First run the cell below to load again some our favourite libraries as well as the data from the first session.

#Keep cell
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import sca

wines_normalized_df =  pd.read_pickle('data/wines_normalized_df.pkl')

One of the constraints is that we need all data to be numerical including the target column 'quality'. We can use our old friend np.where() to do this easily. Run:
```
wines_normalized_df['quality'] = np.where(wines_normalized_df['quality'] == 'bad', 0, 1)
wines_normalized_df.head()
```

0 stands therefore for bad wine and 1 for good wine.

wines_normalized_df['quality'] = np.where(wines_normalized_df['quality'] == 'bad', 0, 1)
wines_normalized_df.head()

The next cell loads the necessary libraries from Keras. There are quite a lot of options, which we can ignore for the time being. Please, just run the cell.

# Keep cell
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical

Like before, we get the x and y values and split the dataset into a train and test set. There is an even more powerful option with scikit's train_test_split function. First import it with `from sklearn.model_selection import train_test_split`.

from sklearn.model_selection import train_test_split

To perfrom the split, we need again X and y for  input and target. It's the same procedure as before. So, please run:
```
X = wines_normalized_df.loc[:, wines_normalized_df.columns != 'quality'].values
y = wines_normalized_df['quality'].values
```

X = wines_normalized_df.loc[:, wines_normalized_df.columns != 'quality'].values
y = wines_normalized_df['quality'].values

Now, we create the test and train data with train_test_split(). Run `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)` to create the split. The function takes the input X and output Y as well as the size of the test data - in this case 25%.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

We have to do one more data preparation step. Our output variable needs to be one-hot encoded. Let's run the function first and then discuss the result to understand what's going on. Type in `y_train_cat = to_categorical(y_train)`. Also print out y_train.

y_train_cat = to_categorical(y_train)
y_train_cat

As you can see, this created a binary representation of the output. So, if it was bad wine, the output would be 1,0 and for good wine it would be 0,1. Why do we need to do this? Otherwise, Keras would interpret the quality column as the numbers 0 and 1. A one hot encoding is a representation of categorical variables as binary vectors. Check out https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/. 

Can you now create y_test_cat in the same way but for y_test?

y_test_cat = to_categorical(y_test)

We are ready to model, as the data is prepared. This is not so very different from what we know from before.

Keras makes it easier for us to create very complex models. Check out the description at https://www.kdnuggets.com/2018/06/keras-4-step-workflow.html. Modern AI seems often to be all about creating every more complext neural network models. Keras is one of the libraries at the heart of this research. 

We will choose to make it fairly easy and create a sequence of neural network layers with Keras Sequential: https://keras.io/guides/sequential_model/. Start with `model = Sequential()`.

model = Sequential()

Following https://www.kdnuggets.com/2018/06/basic-keras-neural-network-sequential-model.html, we need to first define the input layer, which takes all our 11 numerical features. In Keras, this done with the add() function. We want the function to create a fully connected network with Dense() with 50 nodes in the first layer. 

You might remember that we said that these neurons need to fire at each other when activated. activation provides the function to tell the layer when to fire its neurons. 'relu' has become the default option. If you are interested, check https://towardsdatascience.com/7-popular-activation-functions-you-should-know-in-deep-learning-and-how-to-use-them-with-keras-and-27b4d838dfe6. 

Finally, we also need to provide the input shape, which are the number of features we send to the model. In our case these are 11, but we can also get the directly with X_train.shape[1]. 

So, run `model.add(Dense(50, activation='relu', input_dim=X_train.shape[1]))`. 

model.add(Dense(50, activation='relu', input_dim=X_train.shape[1]))

In the next cell, we add two more fully connected layers, one with 25 nodes and one with 5 - because we can. The choice of neural network architectures is hotly debated and part of intense research. Keras makes it easy to add more layers with add(). Run:
```
model.add(Dense(25, activation='relu'))
model.add(Dense(5, activation='relu'))
```

model.add(Dense(25, activation='relu'))
model.add(Dense(5, activation='relu'))

Our final output layer is slighlty different. First the number of ouptut nodes is given by the 2 quality categories we have. We get this with y_train_cat.shape[1]. The activation is very differnt. Softmax is a mathematical function that converts numbers into probabilities (https://machinelearningmastery.com/softmax-activation-function-with-python/). In our case, these are the probabilities for each case - predicting either good or bad wine. Type in `model.add(Dense(y_train_cat.shape[1], activation='softmax'))`.

model.add(Dense(y_train_cat.shape[1], activation='softmax'))

All we have to do now is to compile the model with 
```
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
```

The metrics should be accuracy, which we know already. The optimizer is used to change the weights of the model during its learning phase. adam has become a bit of a standard default option here. loss determines how the machine should calculate the difference between what it has already achieved and how far it is still away from its prediction target. The optimizer tries to minimize this loss. categorical_crossentropy is a standard loss function for classifications - in our case between good and bad wine. 

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

With `model.summary()`, we can print out the model structure we created. 

model.summary()

Now, fit the model like before with `model.fit(X_train, y_train_cat, epochs=10)`. An epoch means the training cycle of a neural network. A forward pass and a backward pass together are counted as one pass. Each time we try to update and improve all the weights in the model based on the feedback from trying to match the current model output with the expected output.

model.fit(X_train, y_train_cat, epochs=10)

As before, we run predict() to get model's predictions of the wines' qualities. Run `y_pred_train = model.predict(X_train)`. This time we will do this for both training and test data.

y_pred_train = model.predict(X_train)

Can you do the same for y_pred_test?

y_pred_test = model.predict(X_test)

We do this for both data set, as we want to avoid not just a model that underfits the data and is therefore bad at predicting new data, which we test with y_pred_train. We also want avoid a model that overfits the data by being great at the training data but not good with the new test data. For a detailed explanation, check https://en.wikipedia.org/wiki/Overfitting. 

To evaluate, we need the accuracy next. Let's start with the one from y_pred_train and take a look at it first. Run ` y_pred_train`

y_pred_train 

You should see an array of two columns, one for each wine entry. My first row, e.g., shows [0.13016923, 0.8698308 ]. Yours might well look differently. We can translate this into: The probabillity of the first row being a bad wine (=0) is 0.13. It is a good wine (=1) with 0.87. 

We can use np.argmax() to ask the machine to return the best prediction for each row - the one with the highest probablity. Run `y_pred_train = np.argmax(y_pred_train, axis=1)` to choose the index of the maximum value for each row, which we say is the best prediction. Why is the index the right answer? 

Also print out y_pred_train.

y_pred_train = np.argmax(y_pred_train, axis=1)
y_pred_train

Success! We have predicted the training data wines. Can you repeat the same for the test data and create y_pred_test?

y_pred_test = np.argmax(y_pred_test, axis=1)
y_pred_test

SciKit has a function to calculate the accuracy score. First import `from sklearn.metrics import accuracy_score`.

from sklearn.metrics import accuracy_score

Run `accuracy_score(y_train, y_pred_train)`.

accuracy_score(y_train, y_pred_train)

Can you do the same for y_test? Of course, you can ...

accuracy_score(y_test, y_pred_test)

Hopefully, these two are not too far apart for you. The model does neither underfit not overfit. It is, however, not  better than the earlier one despite added complexity. The data is obviously not good and big enough. 

That's it. You are now part of the deep learning elite. You could go back and change the model by adding layers or nodes, changing the optimizer, etc. There are a number of options, which you can all experiment with. 

We would like to move on to decision boundaries.

## Decision Boundaries

As a final experiment, let's try and generate decision boundaries in the feature space, which decide in our case about the wines and whether they are good or bad. This is not so easy because we need to reduce the dimensions. So, most of the following code is given.

We need to map our 11 features into 2 features, because we cannot visualise an 11-dimensional space that humans could read. A standard strategy for reducing the dimensions is to apply Principal Component Analysis (PCA) (https://en.wikipedia.org/wiki/Principal_component_analysis). It is a fairly complicated method. If you are interested, check out https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python. Here, it is enough to know that for each wine observation it will find its position in a lower-dimensional feature space. 

Run the cell below. 

Tip: PCA is an important machine learning method, and you might want to try and see whether you can get the top 3 principal components, too. Just to play and learn ...

#Keep cell
from sklearn.decomposition import PCA

pca_no = 2

pca = PCA(n_components = pca_no)
principalComponents = pca.fit_transform(X)
principal_df = pd.DataFrame(data = principalComponents, columns = ['pca_1', 'pca_2'])
principal_df['quality'] = wines_normalized_df['quality']
principal_df.head()

You should see a data frame principal_df. The first two columns are the two prinicipal components and the final the quality. We only use the first 2 principal components, because we would like to create a 2-dimensional feature space readable to humans. 

Next, we will visualise the decision boundaries. Our strategy will be to calculate the wine quality prediction for each point in the 2-dimensional PCA feature space. Because we only have two input dimensions, we cannot use the same model as before. This means we have to run all the steps again. It will be a very good exercise to see whether you understand all the steps.

Run the next cell to create X and y again. We do not have to split the data into test and training, as we want to map all of it. 

#Keep cell
X = principal_df.loc[:, principal_df.columns != 'quality'].values
y = principal_df['quality'].values

y = to_categorical(y)

The following cell contains our model. The only difference is that input_dim = pca_no (2) because we only keep two features - both principal components.

#Keep cell

model = Sequential()
model.add(Dense(50, activation='relu', input_dim=pca_no))
model.add(Dense(25, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

Fit the model ...

#Keep cell
model.fit(X, y, epochs=10)

We create the actual predictions with argmax.

#Keep cell
y = np.argmax(y, axis = 1)
y

We have added a function plot_decision_boundary to the sca libary. Run it with `sca.plot_decision_boundary(X, y, model)`.

sca.plot_decision_boundary(X, y, model)

This might have taken a while because it creates predictions for all points in the space!

The predictions are not great - also because we mapped them into two dimensions -, but we can clearly see that the boundary is a complex function. 

