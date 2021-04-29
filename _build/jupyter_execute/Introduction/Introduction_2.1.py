# Introduction to Python 2.1

# introductions to interactive data analysis with Python
# These should be four sessions to introduce Python and Pandas with some visualisation


In this assignment we will explore and visualise real lige datasets using Python and Pandas
Before we can explore real life datasets, we need to, however, first discuss some more advanced Python constructs.

--> Lists? --> arrays
--> Is matrix needed?



Explanation arrays

First import numpy

Let's first create a vector by creating an array from 1 to 9 np.array(range(1,10)


import numpy as np
import pandas as pd

vec = np.array(range(1,10))

We will use the mtcars dataset. As the name tells you, it contains observations about cars. You can look at the first few observations/rows by typing in head(df). Contemplate a bit how cars are described here. The data frame is a good example of using features to describe observations.

import statsmodels.api as sm

mtcars = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data
df = pd.DataFrame(mtcars)

We will use the mtcars dataset. As the name tells you, it contains observations about cars. You can look at the first few observations/rows by typing in df.head(). Contemplate a bit how cars are described here. The data frame is a good example of using features to describe observations.

df.head()

Now select the first 10 rows of mtcars. Assign it to my_df

my_df = df.head(10)

Finally, let’s collect the list, the matrix and the data frame in a list with my_list = [llist, my_df]

my_list = [vec, my_df]

Print out my_list

my_list

You now have a big collector with my_list! If you would like to access any element, you can use the double square brackets. For the first one, try my_list[[0]]

my_list[0]

Let's create a vector containing months=np.array(['March', 'April', 'January', “'November', 'January', 'September', 'October', 'September', 'November', 'August', 'January', 'November', 'November', 'February', 'May', 'August', 'July', 'December', 'August', 'August', 'September', 'November', 'February', 'April']).



# No option in python to store array in factors
months=np.array(['March', 'April', 'January', 'November', 'January', 'September', 'October', 'September', 'November', 'August', 'January', 'November', 'November', 'February', 'May', 'August', 'July', 'December', 'August', 'August', 'September', 'November', 'February', 'April']) 

months

Python Counter

from collections import Counter

Counter(months)

loops and control structures

indenting


Control structures execute a piece of code based on a condition. You can recognize them in most programming languages by the keyword if. In python they look like the following


medium = 'LinkedIn'

if medium == 'LinkedIn':
    print('Showing LinkedIn information')

Now, also assign num_views = 14.

num_views = 14

Let’s try to confirm whether we are popular with if num_views > 15 print 'You are very popular!'

if num_views > 15:
    print('you are very popular!')

You can combine both expressions also logically. and stands for a logical and, while or stands for the logical or. Try the logical and and type in if (num_views > 15 & medium == 'LinkedIn') print('You are popular on LinkedIn!') 

Explain the results!

if (num_views > 15 and medium == 'LinkedIn'):
    print('You are popular on LinkedIn!')

Finally, you can also tell the computer to do other work if the condition is not fulfilled. You need to use the keyword else

if (num_views > 15 and medium == 'LinkedIn'):
    print('You are popular on LinkedIn!')
else:
    print('Try to be more visible!')

for loops

for letter in 'linkedin':
    print(letter)

For loops can be very useful to perform repeated operations on collections of data, which is something we often want to do. So, for instance, loops could be used to get the square root of each element in a vector or we could use them to calculate the average value of a numeric column in a data frame. However numpy has some easier functions to do so for the entire list or array at once (..)

apply function similar to the numpy functions?

Taking the square root of each element by doing to following:

import math

for num in vec:
    sqr = math.sqrt(num)
    print(sqr)

We can replace it by the following shorter code using numpy

np.sqrt(vec)

Let's say we want to take the average of our linkdin and facebook views, using the following data:

linkedin = np.array([16, 9, 13, 5, 2, 17, 14])
facebook = np.array([17, 7, 5, 16, 8, 13, 14])

print(np.average(linkedin))
print(np.average(facebook))

You can also play with numpy's other functions such as np.char.str_len This counts the number of character for each entry in the days vector. 

days = (['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

np.char.str_len(days)

How about np.char.upper(days)?

np.char.upper(days)

For our final apply example, run np.size(days)

np.size(days)

# In thhe original assigment these functions are used for size of a dataframe while numpy doesn't work for that,
# but pandas does, should this be about pandas then instead of numpy? and then on operations for dataframes
# But umpy functions do work on pandas dataframe columns
# Not really a lapply sapply apply in python

Something about built ins in pandas? 

To add 2 to each element in a numpy array we simply type the name of the array + 2

facebook + 2

np.sum([-2,3])

That’s it with regard to the basics of python. Next, we use our knowledge to analyse a real life social dataset about death penalties and explore how easy it is to plot data with python.

We will work through a detailed example of data exploration and analyse the question of racism with regard to death penalties in the USA. Deathpenalty covers judgements of defendants in cases of multiple murders in Florida between 1976 and 1987. The cases all have features that (a) describe whether the death penalty was handed out (where 0 refers to no, 1 to yes), (b) the race of the defendant and (c) the race of the victim (black is referred as 0, white is 1). Check out the description at http://artax.karlin.mff.cuni.cz/r-help/library/catdata/html/deathpenalty.html



df = pd.read_csv('deathpenalty-florida.csv')

By typing in df.head(), we can see the first couple of cases/rows. What types of columns do you see?

df.head()

By entering df.tail(), we can see the last couple of cases/rows.

df.tail()

Next, we will try to ask the data a few simple questions. Sometimes the solution will be given. Otherwise, you will have to find it yourself.

With the function np.mean() you can retrieve the average of each of the columns

np.mean(df)

We can also only do it for the deathpenalty frequency column by selecting it from the dataframe. You can thus answer what the average frequency of judgements would be by entering np.mean(df['Freq']). Try it.

np.mean(df['Freq'])

mean is very useful to understand data. When something is deemed average, it falls somewhere between the extreme ends of the scale. An average student might have marks falling in the middle of his or her classmates; an average weight is neither unusually light nor heavy. An average item is typical, and not too unlike the others in the group. You might think of it as an exemplar.

median is another function like mean that summarizes a whole dataset by delivering a central tendency. Like mean, it identifies a value that falls in the middle of a set of data. median splits the upper 50% of a data from the lower 50%. It thus delivers the value that occurs halfway through an ordered list of values. How do you get the median frequency of judgements?

np.median(df['Freq'])

What is the lowest number of judgements (min)? Assign the value to min_freq, please.

min_freq = np.min(df['Freq'])

What is the highest number of judgements (max)? Assign the value to max_freq, please.

max_freq = np.max(df['Freq'])

What kind of case combinations had the lowest numbers of judgements? We can obtian these using functions that come with the pandas dataframe, to get the three rows with the smallest value for Freq, type df.nsmallest(1, 'Freq'). To create a subset, assign it to subset_small

df.loc[df['Freq']]
df.nsmallest(1, 'Freq')

Ok, then. Which case combinations had the highest number of judgements? You just need to change the function to nlargest

df.nlargest(1, 'Freq')

min and max are measures of the diversity or spread of data. Knowing about the spread provides a sense of the data’s highs and lows, and whether most values are like or unlike the mean and median. The span between the minimum and maximum value is known as the range. 

# Doesnt really exist in python

The range function returns both the minimum and maximum value. With the diff function you could get the absolute difference. Do you know how? The first and third quartiles, Q1 and Q3, refer to the value below or above which one quarter of the values are found. Along with the median (Q2), the quartiles divide a dataset into four portions, each with the same number of values. Check out np.quantile(df['Freq'], [0.25,0.5, 0.75, 1])

np.quantile(df['Freq'], [0.25,0.5, 0.75, 1])

There are two more really useful data exploration functions in Python. The first one is describe. Try df.describe() and describe what is returned.

df.describe()

summary returns all those value you just tried to find yourself! Oh, well. There is another function data(), which returns the structure of a data frame. It is very useful to find out about columns and features of a dataset. Try df.info().

df.info()

hese summary statistics work only with what in statistics is called numerical data, which is basically anything measured in numbers. Alternatively, if data is represented by a set of categories, it is called categorical or nominal. In our dataset, we do not really have exciting categorical data, because of the way the data set is constructed. In Python, we can compare categorical features by counting the values of for example the VictimRace to see how often each race occurs by typing df.value_counts('VictimRace') and you will see that both victim races are equally represented..

df.value_counts('VictimRace')

We are only getting warm. Let’s make this all a little bit more complicated. We want to know how many black or white people received the death penalty. Remember that a black person is represented by 0 and a white by 1. A death penalty was handed out if DeathPenalty equals 1, with 0 otherwise. Let’s first create a new data frame black_and_deathpenalty, which contains black defendants who received the death penalty by usling df.loc[]. 

black_and_deathpenalty = df.loc[(df['DefendantRace'] == 0) & (df['DeathPenalty'] == 1)]

Similarly, we can get the white defendants who received the death penalty. Get the subset and assign it to white_and_deathpenalty.

white_and_deathpenalty = df.loc[(df['DefendantRace'] == 1) & (df['DeathPenalty'] == 1)]

Overall, we want to compare the likelihood of white and black defendants to receive the death penalty. In order to achieve this, the next step is to find out about the overall number of black people who received the death penalty. You can get this with the sum function, which we have already met. Use n_black_deathpenalty 

n_black_deathpenalty = black_and_deathpenalty['Freq'].sum()


Next, find those whites, which were given a death penalty. Assign a new variable n_white_deathpenalty.

n_white_deathpenalty = white_and_deathpenalty['Freq'].sum()

What is therefore the proportion of black people receiving the death penalty? Remember you can get this by dividing the number of black defendants with the death penalty by the total number of defendants with the death penalty.

n_black_deathpenalty / (n_black_deathpenalty + df['Freq'].sum())

That’s quite a low percentage. Do we then not need to worry then about racial biases in the judgements? What other information do we need to come to such a conclusion? Let’s find out next.

Proportionally, how many of the death penalties handed out to black people were for killing a white person? The expression black_and_deathpenalty.loc[(black_and_deathpenalty['VictimRace'] == 1)]['Freq'] / n_black_deathpenalty is a bit complicated but gets the right result

black_and_deathpenalty.loc[(black_and_deathpenalty['VictimRace'] == 1)]['Freq'] / n_black_deathpenalty

Finally, how likely is it that a white person killing a black person will receive the death penalty?

white_and_deathpenalty.loc[(white_and_deathpenalty['VictimRace'] == 0)]['Freq'] / n_white_deathpenalty

Out of 15 death penalties for black people, 73% for for killing a white person. While none of the 53 death penalties for white people in Florida were given for killing a black person. I hope you can see that there are many powerful functions to explore data directly in Python. Another really good way to explore data is not to ask direct questions but to summarize it with graphs and visualisations. Visualisations and graphs are easy to do with Python.



df.plot.bar()

Now, how about df.boxplot('Freq')? What do you see? Ask the Internet!

df.boxplot('Freq')

The boxplot displays the centre and spread in a format that allows you to quickly obtain a sense of behaviour of the data. The median is denoted by the dark line while the box around it stands for the spread. The boxplot shows one outlier. Can you identify it in the dataset deathpenalty?