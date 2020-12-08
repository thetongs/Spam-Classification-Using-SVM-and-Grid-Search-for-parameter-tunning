## Laad libraries
#
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

##  Load dataset
# 
dataframe = pd.read_csv("spam.csv")
print(dataframe.describe())

## Split
# Independent and Dependent variable
x = dataframe["EmailText"]
y = dataframe["Label"]

## Trainind and Testing dataset
#
x_train,y_train = x[0:4457],y[0:4457]
x_test,y_test = x[4457:],y[4457:]

## Features conversion
#
cv = CountVectorizer()  
features = cv.fit_transform(x_train)

## Model
# parameter
tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}

## Grid search for svm model
# Find better parameter
# Tuned it
model = GridSearchCV(svm.SVC(), tuned_parameters)

# Fit
model.fit(features,y_train)

# Check best parameters by grid search
print(model.best_params_)

# Accuracy
print(model.score(cv.transform(x_test),y_test))



