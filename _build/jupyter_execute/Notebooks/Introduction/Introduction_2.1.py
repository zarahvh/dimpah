# Interactive Data Exploration with Python 2.1

In this chapter, we will explore and visualise real-life datasets using Python and Pandas and query questions of racial bias in death sentences. Before we can explore real-life datasets, we need to discuss a couple of more advanced Python constructs.

First, we need load the libraries Pandas and Numpy. You can run more than one statement in a cell. So, type in: 

```
import numpy as np 
import pandas as pd
```

import numpy as np
import pandas as pd

To introduce more Pandas ideas, we will use the famous iris dataset, every data analyst will know. It contains observations about flowers/irises. The details can be found here: https://archive.ics.uci.edu/ml/datasets/iris. 

"
The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. 
Attribute Information:
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
-- Iris Setosa
-- Iris Versicolour
-- Iris Virginica
"

Run the following cell to load iris from a library and assign it to a dataframe called df.

#Keep cell
from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

Do you almost understand what is going on here? You access the sklearn.datasets library and then load the iris dataset before transforming it into a data frame.

You can look at the first ten observations/rows by typing in df with head(). Type in df.head(10)

df.head(10)

Contemplate a bit how the plants are described. Where is the beauty of a flower? The data frame is a good example of using features to describe observations.

Now, select the first 10 rows into a new data frame with my_df = df.head(10).

my_df = df.head(10)

Next, we want to show you how to add an index to a dataframe. An index can be anything that has unique values in Pandas. First, we create such an index as an array.

The Python function range allows us to create a list of integers (a number that can be written without a fractional component). To create an array ind of 10 numbers starting from 0, run ```ind = np.array(range(0, 10))```. In the same cell, also print it out. We will use ind to index the iris data frame.

ind = np.array(range(0, 10))

ind

Let's assign a column called ind to the iris dataframe my_df. Do you remember how? Yes, it is ```my_df['ind'] = ind```. 

In the same cell, also print out my_df. Ignore the warning SettingWithCopyWarning. This is for demonstration purposes only.

my_df['ind'] = ind
my_df

You can now set the ind column to be the index of the data frame, which will make selecting items, slicing, etc. much faster and consistent. 

Type in ```my_df.set_index('ind', inplace=True)```. And print out my_df.

my_df.set_index('ind')
my_df

The earlier warning came as Pandas prefers us to create an index not for a column but the whole data. 

So, let's do this again. This time, we create a new datafame directly with an index. Type in ```my_df2 = pd.DataFrame(df.head(10), index = ind)```. Print my_df2 out.

my_df2 = pd.DataFrame(df.head(10), index = ind)
my_df2

This should look exactly like the start of the original iris data frame df. Pandas actually automatically creates an index when you call pd.DataFrame. You could, however, make any sequence of unique values into an index. Let's give the flowers names and index the my_df2 directly with them.

First we create a numpy array ```flower_names = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])```. Type it in.

flower_names = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

Now, we recreate my_df2 with the flower_names as the index.  

First, we need to add a new colum called names to the data frame with ```my_df2['names'] = flower_names```.

my_df2['names'] = flower_names

Next we can use ```my_df.set_index('names', inPlace = True)``` to update the index of my_df2. Print out my_df2.

my_df2.set_index('names', inplace = True)
my_df2

With the names index, we can directly select rows for different flower names values using .loc function. To select a single name type in ```my_df2.loc['a']```.

my_df2.loc['a']

Let's pick up two flowers. Type in ```my_df2.loc[['c','d']]```.

my_df2.loc[['c','d']]

You can also select certain columns with a second list of column names. Try ```my_df2.loc[['c','d'], ['sepal length (cm)','sepal width (cm)']]```.

my_df2.loc[['c','d'], ['sepal length (cm)','sepal width (cm)']]

You can also select ranges of index labels. Try ```my_df2.loc['c':'f']```.

my_df2.loc['c':'f']

Finally, a logical selection is also possible with an indexed data frame directly --- not just via selectors as previously discussed. 

Try to retrieve all entries with petal width (cm) = 0.2 by typing in ```my_df2.loc[my_df2['petal width (cm)'] == 0.2]```. Yes, you have to repeat the data frame inside the [].

my_df2.loc[my_df2['petal width (cm)'] == 0.2]

Indexing is a big topic in the Pandas world and unfortunately a bit complicated. Check out https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html. The book Python for Data Analysis is also a great reference. Its author Wes McKinney has actually created Pandas.

There is so much more you can do with data frames in Pandas. Just one more function you will use quite a bit: value_counts(). It counts the number of times a value appears in a column. Try it with ```my_df2['petal width (cm)'].value_counts()```.

my_df2['petal width (cm)'].value_counts()

We have almost made it to our real-life datasets. But before that we need to introduce a few more aspects from Python, the programming language, which you might need in data analysis tasks. The first topic are control structures.

Control structures execute a piece of code based on a condition. You can recognize them in most programming languages by the keyword if. In Python, they look like the following:

#Keep cell

medium = 'LinkedIn'

if (medium == 'LinkedIn'):
    print('Showing LinkedIn information')

The first statement creates a variable medium with 'LinkedIn' as a value. The control structure starts with if, then uses a condition to evaluate to a Boolean (medium == 'LinkedIn'), followed by a colon. The statement to execute follows in the next line.  print('Showing LinkedIn information') will print out if the condition is met.

Important: Observe the indentation, which is a way of telling Python that a group of statements belongs together as a block. Blocks can be regarded as the grouping of statements for a specific purpose. Python really does not like if you mess with its indentation. Check https://www.dummies.com/programming/python/how-to-indent-and-dedent-your-python-code/.

Now, also assign ```num_views = 14```.

num_views = 14

Let’s try to confirm whether we are popular with ```if num_views > 15``` then print 'You are very popular!' How do you do it?

if (num_views > 15):
    print('you are very popular!')

You can combine Boolean expressions/conditions also logically. The keyword 'and' stands for a logical and, while 'or' stands for the logical or in Python. 

Try the logical combination and and type in:

```
if (num_views > 15) and (medium == 'LinkedIn'):
    print('You are popular on LinkedIn!')
```

Explain the results!

if (num_views > 15) and (medium == 'LinkedIn'):
    print('You are popular on LinkedIn!')

Finally, you can also tell the computer to do other work if the condition is not fulfilled. You need to use the keyword else. Run the cell below and opserve the indentation ...

#keep cell

if (num_views > 15) and (medium == 'LinkedIn'):
    print('Condition met.')
    print('You are popular on LinkedIn!')
else:
    print('Condition not met.')
    print('Try to be more visible!')

Another important concept in Python data analysis are for-loops. They are used for iterating over a sequence (like a list or even a secquence of letters or a string). The cell below shows the syntax. Run it.

#Keep cell

for letter in 'LinkedIn':
    print(letter)

For-loops can be very useful to perform repeated operations on collections of data, which is something we often want to do. So, for instance, loops could be used to get the square of each element in a list/array. Or, we could use them to calculate the average value of a numeric column in a data frame, etc.

The next cell introduces the ```for x in list``` construct, which is a short form for going through elements in a list. Run the following to square each number in our list ind and observe the code:

```
for i in ind:
    print(i**2)

```

```i**2``` is the power-operator in Python.

for i in ind:
    print(i**2)

However numpy has some easier functions to apply functions to the entire array at once, which is why we often prefer using it over standard Python. 

Run ```np.square(ind)``` and admire the simplicity. 

np.square(ind)

Here, numpy is simply amazing and super fast. 

Let's say we want to take the average of our linkdin and facebook views. First load the social media data again by running the cell below.

#Keep cell

linkedin = np.array([16, 9, 13, 5, 2, 17, 14])
facebook = np.array([17, 7, 5, 16, 8, 13, 14])

Now run ```np.average(linkedin)``` to get the average of the LinkedIn array.

np.average(linkedin)

To add 2 to each element in a numpy array we simply type the name of the array + 2. Try ```facebook + 2```.

facebook + 2

You can also combine these arrays to get one exciting average count for all our social media with np.average(facebook + linkedin).

np.average(facebook + linkedin)

You can also play with numpy's other functions such as np.char.str_len. This counts the number of character for each entry in the flower_names list. 

Run ```np.char.str_len(flower_names)``` and find out that ...

np.char.str_len(flower_names)

The outcome is not very surprising given that we made up the names of each flower and were so lazy to just use one letter each.

What happens if you type in ```np.char.upper(flower_names)```?

np.char.upper(flower_names)

For our final apply example, run ```np.size(flower_names)```.

np.size(flower_names)

Next, we use our knowledge to analyse a real-life social dataset about death penalties. We will work through a detailed example of data exploration and analyse a very small part of the complex question of racism in the USA. 

The data deathpenalty_df covers judgements of defendants in cases of multiple murders in Florida between 1976 and 1987. The cases all have features that (a) describe whether the death penalty was handed out (where 0 refers to no, 1 to yes), (b) the race of the defendant and (c) the race of the victim (black is referred as 0, white is 1). 

Load the dataset by running the cell below. Later on we will discuss in detail how you can load your own data in data frame, but if you are already curious check out: https://www.w3schools.com/python/pandas/pandas_csv.asp.

#Keep cell

deathpenalty_df = pd.read_csv('data/deathpenalty-florida.csv')

By typing in ```deathpenalty_df.head()```, we can see the first couple of cases/rows. What types of columns do you see?

deathpenalty_df.head()

By entering ```deathpenalty_df.tail()```, we can see the last couple of cases/rows.

deathpenalty_df.tail()

Next, we will try to ask the data a few simple questions. Sometimes the solution will be given. Otherwise, you will have to find it yourself.

With the Numpy function np.mean(), you can retrieve the average of a series of numbers.

We want to know the average of the frequency column by selecting it from the dataframe. You can thus answer what the average frequency of judgements would be by entering ```np.mean(deathpenalty_df['Freq'])```. Try it.

np.mean(deathpenalty_df['Freq'])

The numpy mean function is very useful to understand data. When something is deemed average, it falls somewhere between the extreme ends of the scale. An average student might have marks falling in the middle of their classmates; an average weight is neither unusually light nor heavy. An average item is typical, and not too unlike the others in the group. You might think of it as an exemplar.

median is another function like mean that summarizes a whole dataset by delivering a central tendency. Like mean, it identifies a value that falls in the middle of a set of data. median splits the upper 50% of a data from the lower 50%. It thus delivers the value that occurs halfway if we order a list of values. How do you get the median frequency of judgements? Hint: The Numpy function is called np.median.

np.median(deathpenalty_df['Freq'])

There is a significant diffference between the mean and medium value. What does it mean? Check out https://www.datascienceblog.net/post/basic-statistics/mean_vs_median/.

What is the lowest number of judgements (np.min)? Assign the value to min_freq, please. Print out min_freq, too.

min_freq = np.min(deathpenalty_df['Freq'])
min_freq

What is the highest number of judgements (np.max)? Assign the value to max_freq, please. Print out max_freq, too.

max_freq = np.max(deathpenalty_df['Freq'])
max_freq

What kind of case combinations had the lowest numbers of judgements? We could use min_freq and max_freq to select the specific rows from deathpenalty_df. Do you know how? 

Try it for the highest number of judgements (max_freq).

deathpenalty_df[deathpenalty_df['Freq'] == max_freq]

But, this case is so common that we can optain these cases using functions that come with Pandas. To get the row with the smallest value for Freq, type deathpenalty_df.nsmallest(1, 'Freq'). The first argument specifies the numnber of items to retrieve and the second the column.

deathpenalty_df.nsmallest(1, 'Freq')

Onwards ... 

Which case combinations had the highest number of judgements? You just need to change the function to deathpenalty_df.nlargest.

deathpenalty_df.nlargest(1, 'Freq')

This is already very interesting. While whites murdering white is the by far most common case, there were no deapth penalities here.

min and max are measures of the diversity or spread of data. Knowing about the spread provides a sense of the data’s highs and lows, and whether most values are like or unlike the mean and median. The span between the minimum and maximum value is known as the range. Calculate the range in the next cell.

max_freq - min_freq

The data is very spread out ...

To know more about the distribution of data, np.quantile comes in handy. The function defines cut-off points dividing observations. It is the function that helps us calculate the nth quantile of the given data along. 

Check out ```np.quantile(deathpenalty_df['Freq'], [0.25,0.5, 0.75, 1])```. The first argument is the column and the second argument is the list of the percentage cut-off points at 25%, 50%, 75% and 100%.

np.quantile(deathpenalty_df['Freq'], [0.25,0.5, 0.75, 1])

The answer should read that 9.25 as frequency is the 25% cut-off, 26.5 the 50% cut-off, etc.

There are two more really useful data exploration functions in Python. The first one is describe. Try ```deathpenalty_df.describe()``` and see what is returned.

deathpenalty_df.describe()

describes returns the value you just tried to find yourself! Oh, well. 

There is another Pandas function called info(), which returns the structure of a data frame. It is very useful to find out about columns and features of a dataset. Try ```deathpenalty_df.info()```.

deathpenalty_df.info()

Summary statistics work only with what in statistics is called numerical data, which is basically anything measured in numbers. In the info() output, you can see this in the Dtype column, which is int64 (integer). 

Alternatively, if data is represented by a set of categories, it is called categorical or nominal. In Python, we can compare categorical features by counting the values of the VictimRace to see how often each race occurs by typing deathpenalty_df.value_counts('VictimRace'). You will see that both victim races are equally represented.

BTW: 'race' is a highly contested idea, and we should be more careful than the data we you use, which assigns people to two very simplified categories. Discussing such constraint should be part of your own critical data analysis.

deathpenalty_df.value_counts('VictimRace')

We are only getting warm. Let’s make this all a little bit more complicated. 

We want to know how many 'black' or 'white' defandants received the death penalty. 

Remember that a black person is represented by 0 and a white by 1. A death penalty was handed out if the column DeathPenalty equals 1, with 0 otherwise. Let’s first create a new data frame black_and_deathpenalty, which contains black defendants who received the death penalty by usling deathpenalty_df.loc[]. Do you know how? One way is to use a selector. 

But we can also do it directly with ```black_and_deathpenalty = deathpenalty_df.loc[(deathpenalty_df['DefendantRace'] == 0) & (deathpenalty_df['DeathPenalty'] == 1)]```.

black_and_deathpenalty = deathpenalty_df.loc[(deathpenalty_df['DefendantRace'] == 0) & (deathpenalty_df['DeathPenalty'] == 1)]

Similarly, we can get the white defendants who received the death penalty. Get the subset and assign it to white_and_deathpenalty.

white_and_deathpenalty = deathpenalty_df.loc[(deathpenalty_df['DefendantRace'] == 1) & (deathpenalty_df['DeathPenalty'] == 1)]

Overall, we want to compare the likelihood of white and black defendants to receive the death penalty. In order to achieve this, the next step is to find out about the overall number of black people who received the death penalty. You can get this with the sum function, which we have already met and assign it to n_black_deathpenalty. Run ```n_black_deathpenalty = black_and_deathpenalty['Freq'].sum()```. Print out n_black_deathpenalty.

n_black_deathpenalty = black_and_deathpenalty['Freq'].sum()
n_black_deathpenalty

Next, find those whites, which were given a death penalty. Assign to a new variable n_white_deathpenalty.

n_white_deathpenalty = white_and_deathpenalty['Freq'].sum()
n_white_deathpenalty

What is therefore the proportion of black people receiving the death penalty? Remember you can get this by dividing the number of black defendants with the death penalty by the total number of defendants with the death penalty.

n_black_deathpenalty / (n_black_deathpenalty + n_white_deathpenalty)

That’s quite a low percentage. Do we  not need to worry then about racial biases in the judgements? What other information do we need to come to such a conclusion? Let’s find out next.

Proportionally, how many of the death penalties handed out to black people were for killing a white person? 

The expression ```black_and_deathpenalty.loc[(black_and_deathpenalty['VictimRace'] == 1)]['Freq'] / n_black_deathpenalty``` is a bit complicated but gets the right result. Can you explain how?

black_and_deathpenalty.loc[(black_and_deathpenalty['VictimRace'] == 1)]['Freq'] / n_black_deathpenalty

Finally, how likely is it that a white person killing a black person will receive the death penalty? Can you change the last expression? 

white_and_deathpenalty.loc[(white_and_deathpenalty['VictimRace'] == 0)]['Freq'] / n_white_deathpenalty

0 should be the answer though there were 53 death penalities for white people.

Out of 15 death penalties for black people, 73% were for killing a white person. While none of the 53 death penalties for white people in Florida were given for killing a black person. 

I hope you can see that there are many powerful functions to explore data directly in Python. Another really good way to explore data is not to ask direct questions but to summarize it with graphs and visualisations. Visualisations and graphs are easy to do with Python.

Try ```deathpenalty_df.boxplot('Freq')```? What do you see? Ask the Internet!

deathpenalty_df.boxplot('Freq')

The boxplot displays the centre and spread in a format that allows you to quickly obtain a sense of behaviour of the data. The median is denoted by the dark line while the box around it stands for the spread. The boxplot shows one outlier (the top dot). Can you identify it in the dataset?

But this was just a taste of the visualisations that we focus on in the next session.

