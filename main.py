#Thomas Hübscher / 11.02.2021
# program for predicting glass types via decison tree and random forest

import pandas as pd
from sklearn.model_selection import train_test_split

import dataInfo
import preprocess

filePath = 'glass.csv'
X = pd.read_csv(filePath)
print(X.head())

dataInfo.general(X)
dataInfo.missing_value_per_column(X)
dataInfo.colType(X)

y = X['Type'] #(target we want to predict)

X.drop('Type', axis = 1, inplace=True)
print(X.head())


