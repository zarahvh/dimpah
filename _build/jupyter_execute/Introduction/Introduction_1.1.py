# Introduction to Python 1.1

Welcome to your first session on how to use Python. We will have several more sessions like this. Afterwards, you will have a better understanding of processing cultural and social datasets with Python.

Let’s try to add two numbers. Please, type 4 + 3 into the command prompt. Then, press enter.

4 + 3

Now, let’s try 12 – 5.

12 - 5

And, finally, brackets can be used to indicate that operators in the sub-expression take precedence. Try (5 * 3) – 4.

(5*3)-4

These were all numbers. All this is very similar to what you know from calculators. But computers can process many other symbols as well. Other important symbols include ‘strings’, which you know as words. These can be any number of characters like a, b, c, etc. but also -, 1 or & count as characters.

Let’s try this. Please, type ‘Hello World’ into the command prompt. The quotation marks ’’ indicate that this is a string. You could also use “Hello World”.

"Hello World"

What do you see when you type type(‘string’)?

type('string')

There are many more types, which we will cover throughout the course. Booleans are another important one. They evaluate whether an assumption is True or False. Please, type 4 < 5. What do you get back?

4 < 5

Another important concept that discriminates programming languages from calculators are variables. They are basically names for places in the computer’s memory where we can store things. We create variables with the Python assignment operator =. 

Let’s try that and assign the value 5 to the variable my_apples. Please, type my_apples = 5.



my_apples = 5

Well done. Now print out my_apples. Just type my_apples into the command prompt.

my_apples

Now let’s try to assign two variables.

Type my_oranges = 6. You have now created two variables my_apples and my_oranges.

my_oranges = 6

Just like numbers we can add two numerical variables. Please try it with my_apples + my_oranges.

my_apples + my_oranges

We can also assign the result to a new variable my_fruit. Please type my_fruit = my_apples + my_oranges.

my_fruit = my_apples + my_oranges

To check that the new variable exists, please enter my_fruit

my_fruit

But we can only combine variables of the same type. Please assign the string ‘six’ to my_oranges with my_oranges = ‘six’.

my_oranges = 'six'

Now let’s try and ‘add’ my_apples and my_oranges. Type my_apples + my_oranges and you will get an error. 

my_apples + my_oranges

Variables are very important in any programming language. Another key idea is the function. It is basically a predefined set of commands, which you give your own name. On your calculator you can, for instance, use the inv function to get the inverse of a number. Python also has a lot of pre-defined functions like inv. We can access these using the math module. But in any programming language you can also define your own functions. We will come back to this later.

In Python functions are called with arguments in brackets. Please first import math and then type in math.sqrt(9) to get the square root of 9. sqrt is the function name and 9 is the only argument. 

import math
math.sqrt(9)

#  Function with 3 arguments
# Fucntion with different arguments?

This was a lot of stuff for the first lesson. Unfortunately, this is necessary but once you have learned one programming language all of this becomes quite obvious and repetitive. Before we finish we need to learn one more important concept that is specific to Python. With so-called lists, you can collect several elements in the same variable. This is immensely useful as we see later.

Let’s try lists, which store an ordered set of values called elements. A list can contain any number of elements using brackets. Type in numeric_list = [1,,49] to create a numeric list of three numbers and then print it out. 

numeric_list = [1,10,49]

To check that the new list exists, please type numeric_list.

numeric_list

We can also create string/characters list, with string_list = ['abc', 'def', 'ghi']

string_list = ['abc', 'def', 'ghi']

To check that the new vector exists, please type string_list

string_list

# Python lists can contain different elements

Lists are useful to, for instance, hold your poker winnings during the week. You do not play on Saturday and Sunday, but record all the other 5 days, by entering poker_list = [140, -50, 20, -120, 240]. Thank you datacamp.com for this example! An excellent resource to learn R btw, but unfortunately you have to pay for it.

poker_list = [140, -50, 20, -120, 240]

You feel this is your lucky week. So, you play roulette, too. Please record your winnings with roulette_list =[-24, -50, 100, -350, 10].

roulette_list =[-24, -50, 100, -350, 10]

# Python equivalent to names? --> Dict?

Because you prefer it organised, you would now like to name each of the entries. This is possible in Python with dictionaries. Please create a dict by typing names = {}

names = {}
# To create a dictionary from this list we will need to use for loops, maybe too soon?

names_poker = {'Monday': 140,'Tuesday': -50, 'Wednesday': 20, 'Thursday': -120, 'Friday': 240}

names_poker['Monday']

And the same for the roulette winnings

names_roulette = {'Monday': -24,'Tuesday': -50, 'Wednesday': 200, 'Thursday': -350, 'Friday': 10}

# For the addition as well we would need to add a for loop as well as for dividing each value by 100

Next, you are interested in how much you win in poker and roulette per day. You can simply add up all the elements in the vectors. Remember that we can use built-in functions like sqrt? To add up elements in a list we can use the function sum. Try it with sum(poker_list).

sum(poker_list)

And of course we can also do sum(roulette_list).

sum(roulette_list)

In order to get our total winnings per day for the week, we can simply add both vectors with the Python sum function

total_week = sum(poker_list + roulette_list)

Print out total_week.

total_week

We have almost covered everything there is to know about lists. One more important concept is that lists are indexed. Using square brackets, we can select the first, second and third element directly with 0, 1, 2, etc. respectively. To select Monday’s poker winning simply use square brackets [] and type poker_monday = poker_list[0].

poker_monday = poker_list[0]

Print out poker_monday.

poker_monday

You can also select more than one element with the colon operator. In order to select your winnings from Tuesday to Friday, please run roulette_selection_list = roulette_list[1:5].

roulette_selection_list = roulette_list[1:5]

To find out about our roulette winnings from Tuesday to Friday, we can use sum again. Try sum(roulette_list).

sum(roulette_selection_list)

Let’s also set the Wednesday winnings of roulette_selection_vector to 1000 with roulette_selection_list[1] = 1000.

Print out  roulette_selection_list.

 roulette_selection_list

# Sin only works for entire list using numpy or for loop, already want to introduce that here?
import numpy as np

sin = np.arange(1,20,0.1)
sin = np.sin(sin)
sin

import matplotlib.pyplot as plt
plt.plot(sin)