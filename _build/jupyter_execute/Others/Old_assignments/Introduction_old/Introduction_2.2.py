# Introduction to Python 2.2

Those plots did look promising, didn’t they? Let’s start from the beginning and go through basic plots in Python.

To make it a bit more interesting, we return to the LinkedIn and Facebook view numbers. We would like to investigate their relationship. They should be loaded already. Let's get our Linkedin views of the week again.

linkedin = [16, 9, 13, 5, 2, 17, 14]

Let’s look at the Facebook views again. 

facebook = [17, 7, 5, 16, 8, 13, 14]

With the function plt.plot, we simply plot a list value at a certain index. Try plt.plot(linkedin).

import matplotlib.pyplot as plt

plt.plot(linkedin)

That’s ok but not very pretty. Let's make it blue and add dots for the points with plt.plot(linkedin, marker='o', color='blue')

plt.plot(linkedin, marker='o', color='blue')

Please, now type plt.title(‘LinkedIn’, color=‘red’, fontsize=20) to add a red main title of font size 20. Until we start a new plot with the plot function, we will add to the existing one in Python. In this case, we add a title.

plt.plot(linkedin, marker='o', color='blue')
plt.title('LinkedIn', color='red', fontsize=20)

Better. Now, we would like to compare LinkedIn and Facebook views and create a graph containing both. Let’s start again with setting the xlabel to Days and the ylabel to views.

plt.plot(linkedin, marker='o', color='blue')
plt.xlabel('Days')
plt.ylabel('Views')

Let’s add the facebook graph

plt.plot(linkedin, marker='o', color='blue')
plt.plot(facebook, marker='x', color='red')
plt.xlabel('Days')
plt.ylabel('Views')

Add a title: 'LinkedIn-Facebook-week'

plt.plot(linkedin, marker='o', color='blue')
plt.plot(facebook, marker='x', color='red')
plt.xlabel('Days')
plt.ylabel('Views')
plt.title('LinkedIn-Facebook-week', color='red', fontsize=20)

There are many more ways to improve this graph. You can, for instance, add a better x-axis description

plt.plot(linkedin, marker='o', color='blue')
plt.plot(facebook, marker='x', color='red')
plt.xlabel('Days')
plt.ylabel('Views')
plt.title('LinkedIn-Facebook-week', color='red', fontsize=20)
plt.xticks(ticks=[0,1,2,3,4,5,6],labels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])

plt.show()

Finally, let us add a legend in the bottom right corner. This is a bit more complicated and the Internet is definitely your friend here. We need to add the labels to the plots and finally type plt.legend()

plt.plot(linkedin, marker='o', color='blue', label = 'LinkedIn')
plt.plot(facebook, marker='x', color='red', label = 'Facebook')
plt.xlabel('Days')
plt.ylabel('Views')
plt.title('LinkedIn-Facebook-week', color='red', fontsize=20)
plt.xticks(ticks=[0,1,2,3,4,5,6],labels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])

plt.legend()
plt.show()

Much better. You could add this graph already to your presentations. It looks good enough. There are, however, a million ways to improve this even further in Python. If you are interested, just search the web for all the fantastic visualisations people have created with Python. But we will move on to look at how visualisations can be used with a data frame. Remember, data frames are the workhorses of Python, which we use in almost all our data analysis tasks.

First let’s create a simple data frame with pandas, an easy way to do this when using mulitple lists is to first create a dictionary, type d = {'Facebook':facebook, 'LinkedIn': linkedin}

import pandas as pd

d = {'Facebook':facebook, 'LinkedIn': linkedin}

And create the dataframe

df = pd.DataFrame(d)
df

Now, let’s create a simple barplot of facebook views with barplot

df['Facebook'].plot.bar()

And, an advanced version by using the same as above to add the days and a title, see if you can also find how to change the colors of each bar

df['Facebook'].plot.bar(color=['red', 'blue', 'purple', 'green', 'lavender', 'yellow', 'orange'])
plt.title('Facebook')
plt.xticks(ticks=[0,1,2,3,4,5,6],labels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
plt.show()

Finally we can also create a stacked barplot and show both the views of facebook and linkedin using the same function as before but now adding stacked=True in between the brackets

df.plot.bar(stacked=True)
plt.title('Views')
plt.xticks(ticks=[0,1,2,3,4,5,6],labels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
plt.show()