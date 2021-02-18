#Thomas Hübscher / 15.02.2021
# program for predicting glass types via decison tree and random forest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import dataInfo
import plot

filePath = 'glass.csv'
X = pd.read_csv(filePath)
#print(X.head())

dataInfo.general(X)
dataInfo.missing_value_per_column(X)
dataInfo.colType(X)

#plot.target(X)

y = X['Type'] #(target we want to predict)

X.drop('Type', axis = 1, inplace=True)
#print(X.head())

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)


###first model
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
print(confusion_matrix(y_valid , predictions))
print("Standard Random Forrest: ", model.score(X_valid,y_valid))

###model with paramet optimization

#### n_estimators

ns = [10,20,30,40,50,60,70,80,90,100]
scores =[]
for n in ns:
    model =RandomForestClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    print(confusion_matrix(y_valid, predictions))
    score = model.score(X_valid, y_valid)
    scores.append(score)
    print("Random Forrest: n_estimators = ",n, ": ", model.score(X_valid, y_valid))
plot.paramOptimization(scores,ns)


####with grid search

'''
n_estimators = [10,20,30,40,50,60,70,80,90,100]
max_depth = [2,4,6,8,10]
min_samples_split = [2,3,4,5]
min_samples_leaf = [1,2,3,4,5]

param_grid ={'n_estimators': n_estimators,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split,
             'min_samples_leaf': min_samples_leaf}

model = RandomForestClassifier()

grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, verbose = 2, n_jobs = 4)

grid.fit(X_train, y_train)
'''
#print(grid.best_params_)

### use best params
model1 = RandomForestClassifier(n_estimators=40, max_depth=8, min_samples_split=4, min_samples_leaf=3)
model1.fit(X_train, y_train)
print("Hyper1: ", model1.score(X_valid,y_valid))

model2 = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_split=3, min_samples_leaf=1)
model2.fit(X_train, y_train)
print("Hyper2: ", model2.score(X_valid,y_valid))


###algorithm works poorly for the dataset