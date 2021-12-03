# Interactive Data Exploration with Python 1.1

Jupyter Notebooks are a great data exploration and analysis environment. They allow you to use Python more interactively. Python is of course a fully-fledged programming language for writing long chunks of code and complex applications. But in data analysis you often just want to run one step a time and change things fast. Jupyter Notebook’s allow you do just that. These introduction notebooks explore this aspect. 

If you are interested in Python as a programming language, why not try one of the many online environments that help you learn it? We like the Code Academy course: https://www.codecademy.com/learn/learn-python but there are many more.

Welcome to your first session on how to use Python interactively in Jupyter Notebooks. We will have several more sessions like this. Afterwards, you will have a better understanding of processing cultural and social datasets with Python.

Let’s try to add two numbers. Please, type ```4 + 3``` into cell below. Then, press enter.

4 + 3

Now, let’s try ```12 – 5```.

12 - 5

And finally, brackets can be used to indicate that operators in the sub-expression take precedence - just like at school math. Try ```(5 * 3) – 4```.

(5*3) - 4

These were all numbers. All this is very similar to what you know from calculators. But computers can process many other symbols as well. Other important symbols include ‘strings’, which you know as words. These can be any number of characters like a, b, c, etc. but also -, 1 or & count as characters.

Let’s try this ... Please, type ```'Hello World'``` into the Cell's command prompt. The quotation marks indicate that this is a string. You could also use "Hello World".

'Hello World'

What do you see when you enter ```type(‘string’)```?

type('string')

There are many more types, which we will cover throughout the course. Check put: https://realpython.com/python-data-types/

Booleans are another important type. They evaluate whether an assumption is True or False. Please, type ```4 < 5```. What do you get back?

4 < 5

Another important concept that discriminates programming languages from calculators are variables. They are basically names for places in the computer’s memory where we can store things. We create variables with the Python assignment operator =. 

Let’s try that and assign the value 5 to the variable my_apples. Please, type in ```my_apples = 5```.

my_apples = 5

Well done. 

Now print out my_apples. Just type ```my_apples```.

my_apples

Numbers in Python are generally either integers (those without a floating point) or floats (those with a floating point). Check out https://docs.python.org/3/library/stdtypes.html for an overview of all the different numerical types in Python.

Now let’s try to assign two variables.

Type ```my_oranges = 6```.

my_oranges = 6

You have now created two variables my_apples and my_oranges.

Just like numbers we can add two numerical variables. Please try it with ```my_apples + my_oranges```.

my_apples + my_oranges

We can also assign the result to a new variable my_fruit. Please type ```my_fruit = my_apples + my_oranges```.

my_fruit = my_apples + my_oranges

To check that the new variable exists, please enter ```my_fruit```.

my_fruit

But we can only combine variables of the same type. Please assign the string ‘six’ to my_oranges with ```my_oranges = ‘six’```.

my_oranges = 'six'

Now let’s try and ‘add’ my_apples and my_oranges. Type ```my_apples + my_oranges``` and you will get an error. 

my_apples + my_oranges

Variables are very important in any programming language. 

Another key idea is the function. It is  a predefined set of commands. On your calculator you can, for instance, use the inv function to get the inverse of a number. Python also has a lot of pre-defined functions like inv. But in any programming language you can also define your own functions. We will come back to this later.

In Python, functions are called with arguments in brackets. They are often found in extra libraries that are basically collections of functions. More on that later ...

Please run round(9.5) to run the built-in math function round in Python. You should get 10. round is the function name and 9.5 is the only 'argument' as the input into functions is called.

round(9.5)

This was a lot of stuff for the first lesson. Unfortunately, this is necessary but once you have learned one programming language all of this becomes quite obvious and repetitive. Before we finish we need to learn one more important concept in Python. With so-called lists, you can collect several elements in the same variable. This is immensely useful as we will see later.

Let’s try lists, which store an ordered set of values called elements. A list can contain any number of elements using brackets. Type in ```numeric_list = [1, 10, 49]``` to create a numeric list of three numbers.

numeric_list = [1,10,49]

To check that the new list exists, please type ```numeric_list```.

numeric_list

We can also create a string/characters list, with ```string_list = ['abc', 'def', 'ghi']```.

string_list = ['abc', 'def', 'ghi']

To check that the new vector exists, please type ```string_list```.

string_list

Lists are useful to, for instance, hold your poker winnings during the week. You do not play on Saturday and Sunday, but record all the other 5 days, by entering ```poker_list = [140, -50, 20, -120, 240]```. 

Thank you datacamp.com for this example! An excellent resource to learn data science things btw, but unfortunately you have to pay for it.

poker_list = [140, -50, 20, -120, 240]

You feel this is your lucky week. So, you play roulette, too. Please record your winnings with ```roulette_list =[-24, -50, 100, -350, 10]```.

roulette_list =[-24, -50, 100, -350, 10]

Because you prefer it organised, you would now like to name each of the entries. This is possible in Python with dictionaries. Please create an empty dictionary by typing ```names_poker = {}```.

names_poker = {}

To create a dictionary (which is like a named list), simply use the {} and then for each element the key to access the element comes first, then a ‘:’, and finish it with the value. So, to create a dictionary containing all your poker winnings type in ```names_poker = {'Monday': 140,'Tuesday': -50, 'Wednesday': 20, 'Thursday': -120, 'Friday': 240}```.

Dictionaries are further explained at https://www.w3schools.com/python/python_dictionaries.asp. Also a good resource to learn some more Python.

names_poker = {'Monday': 140,'Tuesday': -50, 'Wednesday': 20, 'Thursday': -120, 'Friday': 240}

Now you can access a single element with ```names_poker['Monday']```. Type it in.

names_poker['Monday']

Next, you are interested in how much you win in poker and roulette per day. 

You can simply add up all the elements in the list. Remember that we can use built-in functions? To add up elements in a list we can use the function sum, which comes directly with Python. Try it with ```sum(poker_list)```.

sum(poker_list)

And of course we can also do ```sum(roulette_list)```.

sum(roulette_list)

In order to get our total winnings per day for the week, we can simply add both vectors with the Python sum function. Type in ```total_week = sum(poker_list) + sum(roulette_list)```.

total_week = sum(poker_list) + sum(roulette_list)

Print out total_week.

total_week

We have almost covered everything there is to know about lists. One more important concept is that lists are indexed. Using square brackets, we can select the first, second and third element directly with 0, 1, 2, etc. respectively. So, the index count in Python starts with 0 for the first element. The last element is then the length of the list n – 1. 

To select Monday’s poker winning (first day of the week) simply use square brackets [] and type ```poker_monday = poker_list[0]```.

poker_monday = poker_list[0]

Print out poker_monday.

poker_monday

You can also select more than one element with the colon operator. 

In order to select your winnings from Tuesday to Friday, please run ```roulette_selection_list = roulette_list[1:5]```.  The first value, left of the colon, is the first index to select (in this case Tuesday). The last value is the first index NOT to select (here an imaginary Saturday). Print out roulette_selection_list.

roulette_selection_list = roulette_list[1:5]
roulette_selection_list

This is called list slicing in Python and requires some getting used to.

If you have a list l, these are the options:

- l[start:stop]  # items start through stop-1
- l[start:]      # items start through the rest of the array
- l[:stop]       # items from the beginning through stop-1
- l[:]           # a copy of the whole array

To find out about our roulette winnings from Tuesday to Friday, we can use sum again. Try ```sum(roulette_selection_list)```.

sum(roulette_selection_list)

Using the index we can also update elements in lists. 

Let’s  set the Wednesday winnings of roulette_selection_vector to 1000 with ```roulette_selection_list[1] = 1000```.

roulette_selection_list[1] = 1000

Print out  roulette_selection_list.

 roulette_selection_list

Well done. Your first lesson is done.

