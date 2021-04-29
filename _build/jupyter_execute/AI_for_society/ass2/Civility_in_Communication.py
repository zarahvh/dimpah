# Civility in Communication

The focus of this assignment will be on a) training a classifier to perform hate speech detection; b) use LIME to explain the classifier's behaviour; c) establish whether the classifier might be biased wrt. different demographic dialects.

This assignment is divided into three parts:
1. **Before the laboratory** (individually): read [LIME's paper](https://arxiv.org/abs/1602.04938) and understand how its Python implementation works: https://github.com/marcotcr/lime (docs: https://lime-ml.readthedocs.io/en/latest/index.html). Check these tutorials in particular: [1](https://marcotcr.github.io/lime/tutorials/Lime%20-%20basic%20usage%2C%20two%20class%20case.html) and [2](https://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html). Furthermore, download the dataset, read its description below and make sure you understand it. Finally, implement a classifier to detect offensive language (use the "label" column in the train and dev datasets). You could for example use a TF-IDF model with any classifier you like from sklearn. Your focus, before the laboratory, is to clearly understand LIME and the proposed dataset, as well as to bring your own classifier to the laboratory.
2. **During the laboratory** (in groups): compare your classifiers and chose one or two to work with (e.g., select the best-performing ones, or those using different methods). Split into two sub-groups: one will use LIME to come-up with explanations for classifications. In particular, they will focus on missclassifications and try to explain those. Another group will select a definition of bias (from week 1) and verify whether your classifier(s) are biased wrt. different demgraphic dialects. For this task, use your classifier(s) on the “mini_demographic_dev.tsv” dataset, and assess bias by demographic group (see below for details). At the end of the laboratory, try to combine your work by using LIME to explain biased classifications.
3. **After the laboratory** (in groups): wrap-up your work and write up your results and thoughts into a brief project report. Make sure to discuss the question of whether you think LIME is effective at explaining your classifier(s), whether you found bias in the classifier, and whether LIME explains biased classifications well (or not).

## Dataset

*This dataset and text is taken with permission from the [Computational Ethics for NLP course, HW2](http://demo.clab.cs.cmu.edu/ethical_nlp2020/homeworks/hw2/hw2.html).*

The primary data for this assignment is available in the dataset folder. **Please note that the data contains offensive or sensitive content, including profanity and racial slurs.**

We provide data drawn from two sources. The first (files "train.tsv" and "dev.tsv") consists of tweets annotated for offensiveness taken from the [2019 SemEval task](https://competitions.codalab.org/competitions/20011) on offensive language detection. In the files "train.tsv" and "dev.tsv", the first column (text) contains the text of a tweet, the second column (label) contains an offensiveness label:

* (NOT) Not Offensive - This post does not contain offense or profanity.
* (OFF) Offensive - This post contains offensive language or a targeted (veiled or direct) offense

The file “offenseval-annotation.txt” provides additional details on the annotation scheme.

We additionally provide a data set of tweets proxy-labelled for race in the file titled “mini_demographic_dev.tsv”. This data is taken from the [TwitterAAE](http://slanglab.cs.umass.edu/TwitterAAE/) data set and uses posterior proportions of demographic topics as a proxy for racial dialect ([details](https://www.aclweb.org/anthology/D16-1120.pdf)). The first column (“text”) contains the text of the tweet, and the second column (“demographic”) contains a label: “AA” (for “African American”), “White”, “Hispanic”, or “Other”. For this assignment, we assume that no tweet in the TwitterAAE data set contains toxic language. Thus, any tweet in this file that is classified as toxic is a false positive.

Finally, both development sets (“dev.tsv” and “mini_demographic_dev.tsv”) contain a column “perspective_score”, which contains a toxicity score. These scores were obtain using the [PerspectiveAPI tool](https://www.perspectiveapi.com/) released by Alphabet. This tool is intended to help “developers and publishers…give realtime feedback to commenters or help moderators do their job”

In all data sets, user mentions have been replaced with the token @USER.

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
import nltk, sklearn

### Load and inspect the dataset

df_train = pd.read_csv("dataset/civility_data/train.tsv", sep="\t")
df_dev = pd.read_csv("dataset/civility_data/dev.tsv", sep="\t")
df_test = pd.read_csv("dataset/civility_data/test.tsv", sep="\t")
df_demo = pd.read_csv("dataset/civility_data/mini_demographic_dev.tsv", sep="\t")

print(df_train.shape)

print(df_train)

df_demo.head()

## Train a classifier

# Your code here (zoals te vinden is in de tutorials die gegeven zijn)

## Explain with LIME

# Your code here

## A biased classifier?

# Your code here