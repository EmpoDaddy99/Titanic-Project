import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# dropping unhelpful columns and replacing object values with dummies
# print(train.isnull().sum()/train.isnull().count())
# print(test.isnull().sum()/test.isnull().count())
train = train.drop(columns = ['Cabin', 'Name', 'Ticket'])
test = test.drop(columns = ['Cabin', 'Name', 'Ticket'])
train = pd.get_dummies(data = train, drop_first = True)
train['Embarked_C'] = 0
train.loc[train['Embarked_S'] + train['Embarked_Q'] == 0, ['Embarked_C']] = 1
test = pd.get_dummies(data = test, drop_first = True)
test['Embarked_C'] = 0
test.loc[test['Embarked_S'] + test['Embarked_Q'] == 0, ['Embarked_C']] = 1
# fill in training values for age and fare
linReg = LinearRegression()
train_not_null = train.dropna(subset = ['Age'])
train_null = train.drop(train_not_null.index)
linReg.fit(train_not_null[['Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']],
		   train_not_null['Age'])
yhat = linReg.predict(
	train_null[['Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']])
train.loc[train['Age'].isnull(), ['Age']] = yhat
# fill in testing values for age
linReg = LinearRegression()
test_not_null = test.dropna(subset = ['Age'])
test_null = test.drop(test_not_null.index)
linReg.fit(test_not_null[['Pclass', 'SibSp', 'Parch', 'Sex_male', 'Embarked_Q', 'Embarked_S']], test_not_null['Age'])
yhat = linReg.predict(test_null[['Pclass', 'SibSp', 'Parch', 'Sex_male', 'Embarked_Q', 'Embarked_S']])
test.loc[test['Age'].isnull(), ['Age']] = yhat
# fill in testing values for fare
linReg = LinearRegression()
test_not_null = test.dropna(subset = ['Fare'])
test_null = test.drop(test_not_null.index)
linReg.fit(test_not_null[['Pclass', 'SibSp', 'Parch', 'Sex_male', 'Embarked_Q', 'Embarked_S']], test_not_null['Fare'])
yhat = linReg.predict(test_null[['Pclass', 'SibSp', 'Parch', 'Sex_male', 'Embarked_Q', 'Embarked_S']])
test.loc[test['Fare'].isnull(), ['Fare']] = yhat
# would typically get rid of outliers, but they're hard to find in this set
scaler = StandardScaler()
train[['Pclass', 'Fare']] = scaler.fit_transform(train[['Pclass', 'Fare']])
print(train.corr()['Survived'])
# Test train split
X = train[['Pclass', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Embarked_C']]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)
# Logistic Regression
logReg = LogisticRegression(solver = 'liblinear')
logReg.fit(X_train, y_train)
logReg_yhat = logReg.predict(X_test)
# K Neighbors Classifier
knc = KNeighborsClassifier(n_neighbors = 3)
knc.fit(X_train, y_train)
knc_yhat = knc.predict(X_test)
# SVC
svc = SVC()
svc.fit(X_train, y_train)
svc_yhat = svc.predict(X_test)
# Decision Tree
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_yhat = dtc.predict(X_test)
# accuracies:
print('Logistic Regression:')
print('Accuracy Score:', accuracy_score(y_test, logReg_yhat))
print('F1 Score:', f1_score(y_test, logReg_yhat))
print('Jaccard Score:', jaccard_score(y_test, logReg_yhat))

print('K Neighbors:')
print('Accuracy Score:', accuracy_score(y_test, knc_yhat))
print('F1 Score:', f1_score(y_test, knc_yhat))
print('Jaccard Score:', jaccard_score(y_test, knc_yhat))

print('SVC:')
print('Accuracy Score:', accuracy_score(y_test, svc_yhat))
print('F1 Score:', f1_score(y_test, svc_yhat))
print('Jaccard Score:', jaccard_score(y_test, svc_yhat))

print('DTC:')
print('Accuracy Score:', accuracy_score(y_test, dtc_yhat))
print('F1 Score:', f1_score(y_test, dtc_yhat))
print('Jaccard Score:', jaccard_score(y_test, dtc_yhat))

# apply best classifier
scaler = StandardScaler()
train[['Pclass', 'Fare']] = scaler.fit_transform(train[['Pclass', 'Fare']])
X_test0 = test[['Pclass', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Embarked_C']]
dtc = KNeighborsClassifier(n_neighbors = 3)
dtc.fit(X, y)
dtc_yhat = dtc.predict(X_test0)
yhat_df = pd.DataFrame(dtc_yhat).round(0)
yhat_df.columns = ['Survived']
yhat_df.index.names = ['PassengerId']
yhat_df.index = yhat_df.index + 892
yhat_df.to_csv('predict.csv')
