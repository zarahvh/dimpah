# Machine Bias

The focus of this assignment will be on replicating and discussing the analysis of the **COMPAS recidivism algorithm** (Correctional Offender Management Profiling for Alternative Sanctions) made by ProPublica. See the [original article](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) for reference.

This assignment is divided into three parts:
1. **Before the laboratory** (individually): start by reading the [original article](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) and the [methodology](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm) used by the authors, and then perform the analysis proposed in Section 1 below. Your focus, before the laboratory, is to clearly understand the dataset and what it contains, as well as the claims made by ProPublica.
2. **During the laboratory** (in groups): delve deeper into the 'fairness' and 'bias' aspects of this story. What do you think the authors mean by 'bias'? Which notion of 'fairness' they imply? Do you think their analysis supports their conclusions? Replicate (parts of) the original analysis and discuss results within your group, as detailed in Section 2 below. Make sure you refer to the [methodology](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm) as you do this.
3. **After the laboratory** (in groups): finish your replication and read critical discussions of ProPublica's work, found in [1](http://www.crj.org/assets/2017/07/9_Machine_bias_rejoinder.pdf), [2](https://arxiv.org/abs/1610.07524), [3](https://arxiv.org/abs/1811.10154), [4](https://hdsr.mitpress.mit.edu/pub/7z10o269/release/4) or [5](https://arxiv.org/abs/1701.08230), among others. Write up your results and thoughts into a brief project report. Make sure to discuss the question of whether you think predictive systems should be used for this kind of screening activities at all, and if so when should you would consider them fair or *not* 'biased'.

> Sections of text marked as this one contain direct quotes from the [original article](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) or [methodology](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm).

***Note: ProPublica's study has been criticized as flawed on several aspects. It is important that you maintain a critical outlook when replicating it, and read up such criticism when you are done with replication (step 3 above).***

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

## 1- Descriptive analysis of the dataset

**Please read these excerpts from the [methodology](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm) first.**

> **Data source**: Through a public records request, ProPublica obtained two years worth of COMPAS scores from the Broward County Sheriff’s Office in Florida. We received data for all 18,610 people who were scored in 2013 and 2014.

> **Data filtering**: Because Broward County primarily uses the score to determine whether to release or detain a defendant before his or her trial, we discarded scores that were assessed at parole, probation or other stages in the criminal justice system. That left us with 11,757 people who were assessed at the pretrial stage.

> **COMPAS recidivism scores**: 
> * Each pretrial defendant received at least three COMPAS scores: “Risk of Recidivism,” “Risk of Violence” and “Risk of Failure to Appear.”
> * COMPAS scores for each defendant ranged from 1 to 10, with ten being the highest risk. Scores 1 to 4 were labeled by COMPAS as “Low”; 5 to 7 were labeled “Medium”; and 8 to 10 were labeled “High.”

> **Data enrichment**: 
> * Starting with the database of COMPAS scores, we built a profile of each person’s criminal history, both before and after they were scored. We collected public criminal records from the Broward County Clerk’s Office website through April 1, 2016. On average, defendants in our dataset were not incarcerated for 622.87 days (sd: 329.19).
> * We matched the criminal records to the COMPAS records using a person’s first and last names and date of birth. This is the same technique used in the Broward County COMPAS validation study conducted by researchers at Florida State University in 2010. We downloaded around 80,000 criminal records from the Broward County Clerk’s Office website.
> * To determine race, we used the race classifications used by the Broward County Sheriff’s Office, which identifies defendants as black, white, Hispanic, Asian and Native American. In 343 cases, the race was marked as Other.
> * We also compiled each person’s record of incarceration. We received jail records from the Broward County Sheriff’s Office from January 2013 to April 2016, and we downloaded public incarceration records from the Florida Department of Corrections website.

> **Data evaluation**: We found that sometimes people’s names or dates of birth were incorrectly entered in some records – which led to incorrect matches between an individual’s COMPAS score and his or her criminal records. We attempted to determine how many records were affected. In a random sample of 400 cases, we found an error rate of 3.75 percent (CI: +/- 1.8 percent).

>**How recidivism is defined**: Northpointe defined recidivism as “a finger-printable arrest involving a charge and a filing for any uniform crime reporting (UCR) code.” We interpreted that to mean a criminal offense that resulted in a jail booking and took place after the crime for which the person was COMPAS scored. [..] **For most of our analysis, we defined recidivism as a new arrest within two years.**

### Get the dataset

# Download the dataset

# Create a 'dataset' folder
!mkdir -p dataset
# Remove existing one if needed
!rm -f dataset/compas-scores-two-years.csv*
# Download
!curl 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv' -o dataset/compas-scores-two-years.csv

# Read the dataset

df_raw = pd.read_csv("dataset/compas-scores-two-years.csv")

df_raw.head()

df_raw.shape

df_raw.columns

#### Further data cleaning

>There are a number of reasons remove rows because of missing data:
> * If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
> * We coded the recidivist flag -- `is_recid` -- to be -1 if we could not find a compas case at all.
> * In a similar vein, ordinary traffic offenses -- those with a `c_charge_degree` of 'O' -- will not result in Jail time are removed (only two of them).
> * We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.

#### Columns of interest

We also focus on the most important columns for this analysis, which are:

1. `age`: Age of the defendant; numeric.
2. `age_cat`: Category of Age. It can be < 25, 25-45, >45.
3. `c_charge_degree`: Degree of the crime. It is either M (Misdemeanor), F (Felony), or O (not causing jail time).
4. `c_jail_in`: Time when the defendant was jailed; timestamp.
5. `c_jail_out`: Time when the defendant was released from the jail; timestamp.
6. `days_b_screening_arrest`: Days between the arrest and COMPAS screening.
7. `decile_score`: The COMPAS score predicted by the system. It is between 0-10.
8. `is_recid`: A variable to indicate if recidivism was done by the defendant. It can be 0, 1, -1.
9. `priors_count`: Count of prior crimes committed by the defendant; numeric.
10. `race`: Race of the defendant. It can be 'African-American', 'Caucasian', 'Hispanic', 'Asian', or 'Other'.
11. `score_text`: Category of decile score. It can be Low (1-4), Medium (5-7), and High (8-10).
12. `sex`: Sex of the defendant. It is either 'Male' or 'Female' 
13. `two_year_recid`: A variable to indicate if recidivism was done by the defendant within two years.

# Filter columns

to_keep = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count',
               'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in',
               'c_jail_out']

df_clean = df_raw[to_keep].copy()

print(df_clean.shape)

# Remove petty traffic offenses that do not cause jail time
cond_1 = (df_clean.c_charge_degree != 'O')

# Remove rows which do not have a compas score
cond_2 = (df_clean.score_text != 'N/A') & (df_clean.is_recid != -1)

# Remove cases in which charge date is more than 30 days
cond_3 = (df_clean.days_b_screening_arrest <= 30) & (df_clean.days_b_screening_arrest >= -30)


df_clean = df_clean[(cond_1 & cond_2 & cond_3)]

print(df_clean.shape)

df_clean.head()

### Your turn

Perform a descriptive analysis of the dataset, in particular focusing on the following variables: `age`, `race`, `sex`, and whether recidivism was done by someone (`is_recid`). Assess how the COMPAS scores (`score_text` and `decile_score`) distribute across these variables, and note your thoughts on what you see.

A **descriptive analysis** is an open-ended exploration of a dataset. Useful steps to consider are exploring summary statistics of each variable (min, max, mean, median, data type, etc.); plotting its distribution; compare variables (e.g., check whether they correlate or plot them jointly); repeat the exploration grouping variables (e.g., plotting the distribution of variable y for different sub-groups according to variable y); and so forth. You should make sure to get acquainted with the dataset, as well as to explore in view of your research question.

*It is important that you compare both the COMPAS scores by group, and the predicted vs actual recidivism by group.*

# your code here

## 2 - Replication

A partial replication of these results will suffice, you are welcome and encouraged to dig deeper if you like.

> Our analysis found that:
> 1. Black defendants were often predicted to be at a higher risk of recidivism than they actually were. Our analysis found that black defendants who did not recidivate over a two-year period were nearly twice as likely to be misclassified as higher risk compared to their white counterparts (45 percent vs. 23 percent).
> 2. White defendants were often predicted to be less risky than they were. Our analysis found that white defendants who re-offended within the next two years were mistakenly labeled low risk almost twice as often as black re-offenders (48 percent vs. 28 percent).
> 3. The analysis also showed that even when controlling for prior crimes, future recidivism, age, and gender, black defendants were 45 percent more likely to be assigned higher risk scores than white defendants.
> 4. Black defendants were also twice as likely as white defendants to be misclassified as being a higher risk of violent recidivism. And white violent recidivists were 63 percent more likely to have been misclassified as a low risk of violent recidivism, compared with black violent recidivists.
> 5. The violent recidivism analysis also showed that even when controlling for prior crimes, future recidivism, age, and gender, black defendants were 77 percent more likely to be assigned higher risk scores than white defendants.

Please replicate 1-4 above; 5 is optional but will still yield you bonus points. 

Points 1 and 2 require you to focus on an error analysis (are prediction errors consistent for different groups of people?). Points 3 to 5 can be done using a logistic regression model and interpreting coefficients using marginal effects or odds ratios. It is a good idea to split the group into (at least) two sub-groups: one focusing on 1-2 and one on 3-4. An extra sub-group could focus on bonus points, which can be acquired via 5 or via discussing and assessinng alternative definitions of fairnness to those used by ProPublica.

Hints: consider using [confusion matrices](https://en.wikipedia.org/wiki/Confusion_matrix) for your error analysis, and [statsmodels](https://www.statsmodels.org/stable/index.html) for regression.

# For point 5 above (violent recidivism), you will need to work with another dataset (which uses the same format and should be pre-processed in the same way)
# Download the dataset

# Create a 'dataset' folder
!mkdir -p dataset
# Remove existing one if needed
!rm -f dataset/compas-scores-two-years-violent.csv*
# Download
!curl 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years-violent.csv' -o dataset/compas-scores-two-years-violent.csv

import statsmodels.api as sm

import statsmodels.formula.api as smf

df_copy = df_clean
df_black = df_copy.loc[df_copy['race']=='African-American']
df_white = df_copy.loc[df_copy['race']=='Caucasian']

# Make dictionairy for African-Americans controlled for certain variables
bb = {}
for index, row in df_black.iterrows():
    # Hash function as key for dictionairy
    hashed = repr([df_black['priors_count'].loc[index], df_black['sex'].loc[index], df_black['c_charge_degree'].loc[index], df_black['age'].loc[index], df_black['days_b_screening_arrest'].loc[index], df_black['two_year_recid'].loc[index], df_black['age_cat'].loc[index]])
    if hashed in bb:
        bb[hashed].append(index)
    else:
        bb[hashed] = [index]

# Make dictionairy for Caucasians controlled for certain variables
bw = {}
for index, row in df_white.iterrows():
    # Hash function as key for dictionairy
    hashed = repr([df_white['priors_count'].loc[index], df_white['sex'].loc[index], df_white['c_charge_degree'].loc[index], df_white['age'].loc[index], df_white['days_b_screening_arrest'].loc[index], df_white['two_year_recid'].loc[index], df_white['age_cat'].loc[index]])
    if hashed in bw:
        bw[hashed].append(index)
    else:
        bw[hashed] = [index]

# Make specific dataframes based on indexes
score_w = 0
score_b = 0

for keys in bw.keys():
    if keys in bb.keys():
        # Get list of indexes
        indexes_white = bw[keys]
        indexes_black = bb[keys]
        
        # Loop through indexes
        values_white = [df_white.loc[j].values for j in indexes_white]
        values_black = [df_black.loc[j].values for j in indexes_black]
        
        # Create dataframes out of data from indexes
        df_new_white = pd.DataFrame(values_white, columns = df_clean.keys())
        df_new_black = pd.DataFrame(values_black, columns = df_clean.keys())
        
        # Fit logistic regression on dataframes
        results_white = smf.ols('decile_score ~ sex + age_cat + priors_count + two_year_recid + age + c_charge_degree + days_b_screening_arrest', data=df_new_white).fit()
        results_black = smf.ols('decile_score ~ sex + age_cat + priors_count + two_year_recid + age + c_charge_degree + days_b_screening_arrest', data=df_new_black).fit()

        # Compute results based on predictions of the fit
        if results_black.predict()[0] > 4:
            score_b += 1
        if results_white.predict()[0] > 4:
            score_w += 1
        
print(score_b / score_w)


