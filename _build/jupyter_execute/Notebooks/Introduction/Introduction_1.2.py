# Interactive Data Exploration with Python 1.2

The next new concept is the numpy array. You often use this construct in data analysis. We want to use the numpy array to filter elements in our lists and even bring them together into two dimensions. Up to now we only had one dimension, but why not add one more?

Numpy has been a long-standing toolkit for data analysis. According to https://numpy.org/, it is 'the fundamental package for scientific computing with Python'. It features heavily in what is still a standard introduction in our field: https://wesmckinney.com/pages/book.html. It is a really good book to further work on the topics that are discussed in these introductions.

First, we have to import the library numpy. You always do that in Python with import. In this case, type in ```import numpy```.

import numpy

Let’s move on from our gambling and look at social media. 

We will now introduce an example from social analytics brought to us by dataquest.com, which we will come back to again later in the course. You have a LinkedIn account and a Facebook account and want to find out, which one has more views and is more successful. You collected the views per day for a particular week. 

Type in ```linkedin = numpy.array([16, 9, 13, 5, 2, 17, 14])``` to record the views for LinkedIn.

linkedin = numpy.array([16, 9, 13, 5, 2, 17, 14])

This is how you create a numpy array directly. If you are reminded of standard Python lists, you are right. For now, numpy arrays and lists are basically the same for us. 

Now, please create the list ```facebook_list = [17, 7, 5, 16, 8, 13, 14]```. 

facebook_list = [17, 7, 5, 16, 8, 13, 14]

You can transform lists easily into arrays with ```facebook = numpy.array(facebook_list)```. Try it.

facebook = numpy.array(facebook_list)

Print out facebook to see that it is numpy array. 

facebook

Matrices are another important concept in data analysis. They are two-dimensional arrays. For instance, black and white images are such matrices. They contain a pixel value at the horizontal and vertical dimension. 

We can create a matrix called views by simply combing linkedin and facebook. Type in ```views = numpy.array([linkedin, facebook])```.

views = numpy.array([linkedin, facebook])

Print out views.

views

We can then ask a couple of good questions against the matrix views without having to reference the arrays it is made of. 

To find out on which days we had 13 views for either LinkedIn or Facebook, we type ```views == 13```. The == is the Boolean equivalence operator.

views == 13

When are views less than or equal to 14? Try ```views <= 14```.

views <= 14

How often does facebook equal or exceed linkedin views times two? This is actually a quite advanced expression already. Try it with ```sum(facebook >= (linkedin * 2))```. 

sum(facebook >= (linkedin * 2))

Take a moment to think about the components of this expression. Maybe, you want to take a piece of paper and a pen to write down all the components. Hint: Boolean variables can also be thought of a 0 for False and 1 for True.

Similar to arrays, we can access each element of a matrix, but this time we of course need 2 dimensions. ``views[0][1]`` will select the first row’s second element. Try it.

views[0][1]

The order is of indexs is row first and then column. Try `views[1][4]`.

views[1][4]

The most important structure in Python to store and process data are ‘data frames’. The library to do so is Pandas that is a key part of our work. It is another library just like numpy. With `import pandas as pd`, we tell Python to use the acronym pd for accessing pandas operations. This way we avoid having to type in pandas all the time. 

Try `import pandas as pd`.

import pandas as pd

Just like matrixes, data frames also have rows and columns but can hold different types of variables in each of their columns. 

Think about it for a moment. The power! This way, we can record any observation in the world. Any observation will have different attributes we associate with it. For instance, flowers can be of different types and colours. With data frames we can record each observation of flowers by recording it in rows and assign to the columns the various features we observe like colour, type, etc. This way the whole world is for us to record in data frames!

Let’s return to our social media records of our week and create a data frame to hold all its glorious details.

First, some cleaning up. We want to reorder the social media views, because at the moment the columns are the days of the week and the rows are LinkedIn and Facebook. However, we want the social media companies to be the columns and the rows to be the days. 

If you remember your school days, you can do change the axes in a matrix by transposing it - in Python with numpy.transpose. Simply run `views_t = numpy.transpose(views)`. Print out views_t. 

views_t = numpy.transpose(views)
views_t

In order to get our week together into a single data frame, we create `my_week_df = pd.DataFrame(views_t, columns = ['Facebook', 'LinkedIn'])`. To create my_week_df, first the DataFrame function from Pandas (pd) is called with the transposed views. It is given the columns attribute with a list of names for the columns. 

my_week_df = pd.DataFrame(views_t, columns = ['facebook', 'linkedin'])

Print out my_week_df.

my_week_df

Just like matrixes, we can select rows and columns. For that, we need the operator iloc[row][column]. We need to use the Pandas iloc function here but otherwise it is the same principle as for lists and arrays. 

Select row 1, column 2 with `my_week_df.iloc[0][1]`. Please note, that this function uses [] brackets instead of (). I guess they wanted to give it the numpy feel.

my_week_df.iloc[0][1]

Do you remember how to select multiple elements in a list? 

It is similar for data frames. In order to select row 1 and 2, type in `my_week_df.iloc[0:2][:]`. This means that the first colon operator selects everything between its first index and the second index – 1. The second colon operator uses : to select all columns.

my_week_df.iloc[0:2][:]

You can also leave out the [:]. Run `my_week_df.iloc[1]` to select the second row.

my_week_df.iloc[1]

Any idea how to select the second column rather than a row? Take a look at https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html, and you will know it is `my_week_df.iloc[:, 1]`. Try it.

my_week_df.iloc[:, 1]

Do you know how to select the third and forth row?

my_week_df.iloc[3:5, :]

You can also select the facebook column directly with `my_week_df['facebook']`. 

my_week_df['facebook']

If you use two [[ ]], you get back a new dataframe. Try `my_week_df[['facebook']]`.

my_week_df[['facebook']]

Let's add another column to our dataframe, which we call 'happy' for our happy days in the week.

First define a list `happy = [False, True, False, True, False, True, False]`.

happy = [False, True, False, True, False, True, False]

We can then append this list to the dataframe by naming a new column and assign happy to it. Run ``my_week_df['happy'] = happy``.

my_week_df['happy'] = happy

Print out my_week_df.

my_week_df

Say, we would like to select all days, we were happy. We first define a selector series for the happy days with `happy_days = df['happy'] == True`. Remember the numpy arrays? This is very similar.

happy_days = my_week_df ['happy'] == True

Print out happy_days.

happy_days

It is a 'Series' of Booleans. Series is yet another term for lists and arrays. Pandas calls single columns things like that. I know ...

Now, we use the Boolean selector to create `happy_days_df = df.loc[happy_days]`.

happy_days_df = my_week_df.loc[happy_days]

It looks complicated but it is actually just a combination of statements we already know. Take some time and look at the expression happy_days_df. What kind of parts can you identify? How are the rows/days selected when you were happy?

Print out happy_days_df.

happy_days_df

If you would like to select all the days/rows when you had more views on LinkedIn than Facebook, you can proceed in a similar way. First define the selector with `small = my_week_df['linkedin'] > my_week_df['facebook']`.

small = my_week_df['linkedin'] > my_week_df['facebook']

Then, create a new data frame with `small_df = my_week_df.loc[small]`.

small_df = my_week_df.loc[small]

Print out small_df.

small_df

That’s almost it for data frames. 

I promise they get more interesting once we start working with real data. One more thing you often want to do is to sort a data frame according to one of its columns. We have another Pandas function for that called sort_values. Run `sorted_df = my_week_df.sort_values(by=['facebook'])` to sort my_week_df by the values in the facebook column.

sorted_df = my_week_df.sort_values(by=['facebook'])

Print out sorted_df.

sorted_df

That’s it for the most important concepts around data frames in Python. 

Next, we move on to some real-life datasets.

