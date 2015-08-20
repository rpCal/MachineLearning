#!/usr/bin/python

import pandas
import numpy
import re

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def PreProcessData(Sample, TrainingSample):
	"""Pre-process the loaded data"""
	Sample["Age"] =  Sample["Age"].fillna(TrainingSample["Age"].median())
	Sample["Fare"] = Sample["Fare"].fillna(TrainingSample["Fare"].median())

	Sample.loc[Sample["Sex"] == "male",   "Sex"] = 0
	Sample.loc[Sample["Sex"] == "female", "Sex"] = 1

	Sample["Embarked"]                    = Sample["Embarked"].fillna("S")
	Sample.loc[Sample["Embarked"] == "S", "Embarked"] = 0
	Sample.loc[Sample["Embarked"] == "C", "Embarked"] = 1
	Sample.loc[Sample["Embarked"] == "Q", "Embarked"] = 2

	# Data transformation for better results

	Sample["FamilySize"] = Sample["SibSp"] + Sample["Parch"]
	Sample["NameLength"] = Sample["Name"].apply(lambda x: len(x))

	titles = Sample["Name"].apply(get_title)
	# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 9}

	for k,v in title_mapping.items():
		titles[titles == k] = v

	Sample["Title"] = titles

"""Load the Data"""

titanic_train = pandas.read_csv("train.csv")
titanic_test  = pandas.read_csv("test.csv")
PreProcessData(titanic_train, titanic_train)
PreProcessData(titanic_test, titanic_train)

"""Train the Different Models"""

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic_train[features], titanic_train["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -numpy.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(features)), scores)
plt.xticks(range(len(features)), features, rotation='vertical')
#plt.show()

# Pick only the four best features.
features = ["Pclass", "Sex", "Fare", "Title"]


"""Linear Regression"""

kf = KFold(titanic_train.shape[0], n_folds=3, random_state=1)
predictions = []
alg = LinearRegression()

for train, CVTest in kf:
	train_input      = (titanic_train[features].iloc[train,:])
	train_output     = titanic_train["Survived"].iloc[train]
	alg.fit(train_input, train_output)

	CV_predictions = alg.predict(titanic_train[features].iloc[CVTest,:])
	predictions.append(CV_predictions)

predictions = numpy.concatenate(predictions, axis=0)

predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

i = 0
accuracy = 0

for prediction in predictions:
	if int(prediction) == titanic_train["Survived"][i]:
		accuracy = accuracy + 1

	i = i+1

accuracy = float(accuracy) / len(predictions)

print("Accuracy with LinearRegression :" , accuracy)

"""Logistic Regression"""

alg = LogisticRegression(random_state=1)

LogisticScores = cross_validation.cross_val_score(alg, titanic_train[features], titanic_train["Survived"], cv=3)

print("Accuracy with LogisticRegression :", LogisticScores.mean())

alg.fit(titanic_train[features], titanic_train["Survived"])

predictions = alg.predict(titanic_test[features])

""" Create a submission """
submission = pandas.DataFrame({
	"PassengerId": titanic_test["PassengerId"],
	"Survived"   : predictions
	})

submission.to_csv("kaggle.csv", index=False)

""" Random Forests!"""
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
RandomForestScores = cross_validation.cross_val_score(alg, titanic_train[features], titanic_train["Survived"], cv=3)

print("Accuracy with RandomForest 1:", RandomForestScores.mean())

# Tweeking parameters to account for more efficiency
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
RandomForestScores = cross_validation.cross_val_score(alg, titanic_train[features], titanic_train["Survived"], cv=3)

print("Accuracy with RandomForest 2:", RandomForestScores.mean())


""" Gradient Boosting Classifier"""

# The algorithms we want to ensemble.
# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked","Title"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "Title", "Age", "Embarked"]]
]

# Initialize the cross validation folds
kf = KFold(titanic_train.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_output = titanic_train["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic_train[predictors].iloc[train,:], train_output)
        # Select and predict on the test fold.  
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic_train[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = numpy.concatenate(predictions, axis=0)

i = 0
accuracy = 0

for prediction in predictions:
	if int(prediction) == titanic_train["Survived"][i]:
		accuracy = accuracy + 1

	i = i+1

accuracy = float(accuracy) / len(predictions)

print("Accuracy with GradientBoostingClassifier :" , accuracy)