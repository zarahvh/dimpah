# Churn Analysis

A big topic in digital marketing can be the focus on  customers you can also keep. This is the subject of churn analysis.

Customer churn (most commonly called 'churn') is defined in the article as customers that stop doing business with a company or a service. So, for instance, which costumer will move to a different mobile provider? 

This churn analysis is key to customer relationship management (CRM).

Churn analysis is based on data that contains information on previous customers who churned. We want to compare the current costumers with these to find out whether they might churn, too? Sounds familiar? It's basically a typical case for a decision classification - just like the wine tasting or the Titanic data. 

This notebook will demonstrate to you that we can use the same kind of modelling techniques we used for wines, the titanic, etc. with digital marketing data. Churn analysis is a typical use case of algorithmic decision-making.

## Bank Use Case

A typical example of churn analysis is to find costumers who defaulted on a loan in a bank. Let's load the churn data.

import matplotlib.pyplot as plt
import pandas as pd

churnData = pd.read_csv("churnData.csv")
churnData.describe()

There is a lot of information in this dataset! We are mainly interested in PaymentDefault. PaymentDefault = 1 if the costumer defaulted on the payment and 0 otherwise.

churnData['PaymentDefault'].plot.hist(bins=2)

## Modelling

Let's build a logistic regression model next to find out what in the data helps us predict the default on a loan. It's just another model to predict a target that is categorical.

For example:
- To predict whether an email is spam (1) or (0)
- Whether the tumor is malignant (1) or not (0)

In our case, we use the model to understand which features contribute most to a potential default.

glm stands for Generalized Linear Models with family=binomial indicating a logistic model: https://www.statmethods.net/advstats/glm.html

import statsmodels.api as sm

X = churnData[['limitBal', 'sex', 'education', 'marriage',
                   'age', 'pay1', 'pay2', 'pay3', 'pay4', 'pay5', 'pay6', 'billAmt1', 
                   'billAmt2', 'billAmt3', 'billAmt4', 'billAmt5', 'billAmt6', 'payAmt1', 
                   'payAmt2', 'payAmt3', 'payAmt4', 'payAmt5', 'payAmt6']]
Y = churnData['PaymentDefault']
X = sm.add_constant(X)

model = sm.GLM(Y.astype(float), X.astype(float)).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)

You might have noticed that we have not discriminated test and training datasets to build the model, as we are less interested at this moment in the prediction the model can develop, but in the kind of features which are important to predict default. These are the characteristics in customers we want to watch out for.


According to lm, education is really important to understand default as are sex and the order of payments.

# source: https://planspace.org/20150423-forward_selection_with_statsmodels/
import statsmodels.formula.api as smf
def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

model = forward_selected(churnData, 'PaymentDefault')

print(model.model.formula)

print(model.rsquared_adj)

We are left with pay1 + billAmt1 + pay3 + payAmt1 + marriage + pay5 + education + pay2 + billAmt2 + age + limitBal + payAmt2 + sex + payAmt4 as the important features

## Prediction

Of course we can also use the model to create the predictions we already know about now. Let's try again the full data set first.

Confusionmatrix

X_aic = X[['limitBal', 'sex', 'education', 'marriage',
                   'age', 'pay1', 'pay2', 'pay3', 'pay5', 'billAmt1', 
                   'billAmt2', 'payAmt1',
                   'payAmt2', 'payAmt4']]

X_aic = sm.add_constant(X_aic)
model = sm.GLM(Y, X_aic).fit()

preds = model.predict()

def to_binary(predictions, thresh):
    threshold = thresh
    binary = []
    for item in predictions:
        if item < threshold:
            item = 0
        else:
            item = 1
        binary.append(item)
    return binary

binary = to_binary(preds, 0.5)

from sklearn.metrics import (confusion_matrix,
                           accuracy_score)
cm = confusion_matrix(binary, Y)
acc = accuracy_score(Y, binary)

print('CM:', cm)
print('Accuracy:', acc)

The prediction accuracy is not too bad with 0.80. 

Looking at the confusion matrix, it seems that the model mainly has problems to identify cases where the original value was 0 or no payment default. In ~3519 cases, it assumed that there would be a default. This means that customers would be unnecessarily targeted. In the case of bank defaults, this might upset these costumers. So, we should try and reduce this value.

### Model Improvements by Lowering the Threshold

One way to do this is to introduce a less severe threshold for suspecting defaults on payments. This will avoid customer dissatisfaction. If you check the above formula for prediction we introduced a threshold of 0.5 in the prediction. All values below 0.5 are 0 (= no default) and above 0.5 they are 1 (default).

We can do this, aspredict actually returns a value between 0 and 1. Take a look at the value of the first prediction and you will see a value of ~0.45 - pretty close to the threshold of 0.5.

print(preds[0])

What if try out different threshold, can we improve the prediction by being less strict or maybe we should consider even more cases for payment default and lower the threshold?

We need to make a decision what kind of case we would like to avoid. If you recall our discussion earlier of the confusion matrix, we want to avoid the case where we predicted no default but the default really did happen. We want to reward, however, the case where we predicted no-default and the costumer really do not default.  We can access the confusion matrix results with cm as well as the specific cells in the matrix.

Check it out:

print(cm)
print(cm[0,0])
print(cm[1,0])

It's time to play with the results using the threshold. 

Let's say we define a score function: score <- cm[1,1]*250 + cm[1,1]*250 - cm[1,2]*1000 - cm[1,2]*1000. What does this say? It says that we want to reward the right prediction with a factor of 250 but apply a penalty of -1000 for the wrong prediction. We use a much higher value for the wrong prediction as we are concerned about customer dissatisfaction if we target too many people.

Now, we try a brute-force test using the the a number of threshold between 0.1 and 0.5.

thresholds = [0.1,0.2,0.3,0.4,0.5]

for t in thresholds:
    bin_preds = to_binary(preds, t)
    cm = confusion_matrix(bin_preds, Y)
    score = cm[0,0]*250 + cm[1,1]*250 - cm[1,0]*1000 - cm[0,1]*1000
    print(t, score)
    

Great! That wasn't easy! Take your time going through the details. 

You could see that the optimal threshold is 0.4. It has the best score.

Let's use this threshold and do some real prediction next with a test and training data. 

Following our experience with the wine data, we know that the dataset contains 18,000 entries. Splitting this up into 70-30 training and test data, implies:

from sklearn.model_selection import train_test_split

train, test = train_test_split(churnData, train_size=0.7)

We copy the optimal formula from the AIC test and run:

X_train = train[['limitBal', 'sex', 'education', 'marriage',
                   'age', 'pay1', 'pay2', 'pay3', 'pay5', 'billAmt1', 
                   'billAmt2', 'payAmt1',
                   'payAmt2', 'payAmt4']]
Y_train = train['PaymentDefault']
Y_test = test['PaymentDefault']
X_test = test[['limitBal', 'sex', 'education', 'marriage',
                   'age', 'pay1', 'pay2', 'pay3', 'pay5', 'billAmt1', 
                   'billAmt2', 'payAmt1',
                   'payAmt2', 'payAmt4']]

model = sm.GLM(Y_train, X_train).fit()
Y_pred = model.predict(X_test) 


binary = to_binary(Y_pred, 0.4)

cm = confusion_matrix(binary, Y_test)
acc = accuracy_score(Y_test, binary)
print(cm, acc)

Using our optimal threshold 0.4, we improved the accuracy slightly to 0.804. But we significantly improved the cases where customers wrongly targeted to 814 cases in total.