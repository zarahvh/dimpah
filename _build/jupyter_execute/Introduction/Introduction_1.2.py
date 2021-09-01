# Introduction to Python 1.2

The final new concept for today is the matrix. It has little to do with the movie but is simply a two-dimensional array. Up to now we only had one dimension, but why not add one more?

Let’s try this and move on from your gambling. We will now introduce an example from social analytics brought to us by dataquest.com, which we will come back to again later in the course. You have a LinkedIn account and a Facebook account and want to find out which one has more views and is more successful. You collected the views per day for a particular week in two arrays. Type in linkedin = [16, 9, 13, 5, 2, 17, 14]

import numpy as np

linkedin = np.array([16, 9, 13, 5, 2, 17, 14])

And now create facebook = [17, 7, 5, 16, 8, 13, 14]

facebook = np.array([17, 7, 5, 16, 8, 13, 14])

Python doesn't have a built-in function for matrices. However, we can create a matrix by treating it as a list of a list. Or an numpy array of arrays (using numpy? --> possibility of booleans)

views = np.array([linkedin, 
         facebook])

Print out views.

views

We can then ask a couple of good questions against the matrix views without having to reference the arrays it is made of. To find out on which days we had 13 views for either LinkedIn or Facebook, we type views == 13. The == is the Boolean equivalence operator.

views == 13

When are views less than or equal to 14? Try views <= 14.

views <= 14

How often does facebook equal or exceed linkedin times two? This is actually a quite advanced expression in R already. Try it with sum (facebook >= linkedin * 2). Take a moment to think about the components of this expression. Maybe, you want to take a piece of paper and a pen to write down all the components.

sum(facebook >= (linkedin * 2))

Similar to arrays, we can access each element of a matrix, but this time we of course need 2 dimensions. views[0][1] will select the first row’s second element. Try it.

views[0][1]

Overall, the order is row first and then column. Try views[1][4].

views[1][4]

The most important structure in Python to store and process data are data frames, the most common tool in Python to do so is Pandas. Just like matrixes, data frames also have rows and columns but can hold different types of variables in each of their columns. Think about it! This way we can record any observation in the world. Any observation will have different attributes we associate with it. For instance, flowers can be of different types and colours. With data frames we can record each observation of flowers by recording it in rows and assign to the columns the various features we observe like colour, type, etc. This way the whole world is for us to record in data frames!

Let’s return to the records of our week and create a data frame to hold all its glorious details.

In order to get our week together into a single data frame, we first import Pandas as pd, next 
type in df = pd.DataFrame(views). This combines all our vectors into the data frame my_week_df. Each row is a day in your life.

import pandas as pd

df = pd.DataFrame()

df['linkedin'] = linkedin
df['facebook'] = facebook
df

Just like matrixes, we can select rows and columns with the operator [row][column]. Select row 1, column 2 with df.iloc[0][1]

df.iloc[0][1]

You can also select entire rows/observations using pandas' .iloc, type in df.iloc[1] to select the second row.

df.iloc[1]

Do you remember how to select multiple elements in a vector? It is similar for data frames. In order to select row 1 and 2, type in df.iloc[0:2].

df.iloc[0:2]

Any idea how to select the second column rather than a row? Try it by yourself!

df.iloc[:,1]

Do you know how to select the third and forth row?

df.iloc[3:5,:]

We can also select columns based on their names using df.loc, try selecting facebook

df.loc['facebook']

Let's add another column to our dataframe : happy

First define happy = [False, True, False, True, False, True, False]

We can then append this list to the dataframe by naming a new column and assign happy to it

happy = [False, True, False, True, False, True, False]

df['happy'] = happy

Say we would like to select all days, we were happy. This is easy in python with happy_days_df = df.loc[df['happy']==True]

happy_days_df = df.loc[df['happy']==True]

happy_days_df

Take some time and look at the expression happy_days_df = df.loc[df['happy']==True]. What kind of parts can you identify? How are the rows/days selected when you were happy? 

f you would like to select all the days/rows when you had more views on LinkedIn than Facebook, you can type small_df = df.loc[df['linkedin'] > df['facebook']]

small_df = df.loc[df['linkedin'] > df['facebook']]

small_df

That’s almost it for data frames. I promise they get more interesting once we start working with real datasets. One more thing you often want to do is to sort a data frame according to one or more of its columns. We have another function for that called df.sort_values.

sorted_df = df.sort_values(by=['facebook'])

sorted_df

That’s it for the most important concepts around data frames in Python. Next week we first cover a few things we do not really need for the time being but that are good to know nevertheless. You will need them for reading some of the code you find online. 

...