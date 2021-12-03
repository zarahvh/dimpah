import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from collections import Counter

from matplotlib.figure import Figure


#PredictionModelling2

#https://jonchar.net/notebooks/Artificial-Neural-Network-with-Keras/ changed with labels = np.argmax(labels, axis = 1)

def plot_decision_boundary(X, y, model, steps=1000, cmap='Set1'):
    """
    Function to plot the decision boundary and data points of a model.
    Data points are colored based on their actual label.
    """
    cmap = plt.get_cmap(cmap)
    plt.rcParams["figure.figsize"] = (8,8)
    
    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])
    #Added:
    labels = np.argmax(labels, axis = 1)

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)
    
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, lw=0)
    
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    
    plt.title('Decision Boundary from Deep Learning', size=16)

    
    return fig, ax