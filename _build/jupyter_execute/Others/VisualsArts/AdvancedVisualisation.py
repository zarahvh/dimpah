# Lots of fun with Python visualisations

In this exercise, we use a simple dataset to experience more of the power of matplotlib or seaborn?. It is different from other exercises we have seen so far, as we really focus here on the technology rather than the exploration of real-life data.

## Our dataset today: Iris

This famous (Fisher's or Anderson's) Iris dataset has the measurements in centimeters of the variables sepal.length and width and petal.length and width, respectively, for 50 flowers from each of 3 species of iris. The species are Iris setosa, versicolor, and virginica.

The data was collected by Anderson, Edgar (1935). The irises of the Gaspe Peninsula, Bulletin of the American Iris Society, 59, 2–5.

# size needs to be changed
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('iris.png')
imgplot = plt.imshow(img)
plt.show()

import pandas as pd

iris = pd.read_csv('iris.csv', delimiter=';')

iris.head()

## Grammar of Graphics

Just to repeat what we have learned in today's lecture. The grammar of graphics describes:

- A set of rules for creation of graphics
- Each graphic is made up of several independent components
- Each component can be manipulated
- We can combine components in a specific way to create graphics

To speak the grammar of graphics we need to define:

- Data (noun/subject)
- Aesthetic mappings (adjectives)
- Geom (verb)
- Stat (adverb)
- Position (preposition)

## Seaborn

### Seaborn countplot

# https://www.kaggle.com/zachgold/python-iris-data-visualizations
# plot sepal length count (bar)
import seaborn as sns
import matplotlib.pyplot as plt

fig_dims = (12, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.countplot(x="sepal_length", data=iris, ax=ax)

### Seaborn scatterplot

sns.relplot(x="sepal_length", y="sepal_width", data=iris)

### Using color to represent the species

sns.relplot(x="sepal_length", y="sepal_width", data=iris, hue="species")

### Distribution of species

sns.countplot(x="species", data=iris)

### Line plot 

A line plot does not really makes sense here, because the points are not related/connected but for the same of a demonstration of how to do them with data.

sns.relplot(x="petal_length", y="petal_width", data=iris, hue="species", kind='line')

### Boxplot

sns.boxplot(x='species', y='sepal_length', data=iris)

### Facets

g = sns.FacetGrid(iris,  col="species")
g.map_dataframe(sns.histplot, x="sepal_length")

g = sns.FacetGrid(iris,  row="species")
g.map_dataframe(sns.histplot, x="sepal_length")

### Linear regression line
Use sns.lmplot() to add a linear regression line

sns.lmplot(x="sepal_length", y="sepal_width", data=iris, hue="species", height=10)

## Themes
We can also change the overall look of our theme using sns.set_style()

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.relplot(x="sepal_length", y="sepal_width", data=iris, hue="species")

### Colour scheme
We can easily adjust the colours using the palette parameter

sns.relplot(x="sepal_length", y="sepal_width", data=iris, hue="species", palette="bright")

# interactive plots in python?