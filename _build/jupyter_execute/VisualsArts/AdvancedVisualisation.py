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

# https://www.kaggle.com/zachgold/python-iris-data-visualizations
# plot sepal length count (bar)

# plot sepal width against length (scatter)

# add species with colors

# distribution of species

# line plot with the same

# boxplot for lenght and species

# facets for length?

# themes?

# color schemes?

# shiny app?