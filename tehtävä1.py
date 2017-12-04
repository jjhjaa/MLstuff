import pandas as pd
import numpy as np
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Concrete_Data.csv")
df = df[['Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water', 'Superplasticizer', 'CoarseAggregate', 'FineAggregate', 'Age', 'Strength']]
#print(dff.head(10))

forecast_col = 'Strength'
df['label'] = df[forecast_col]
#print(df.head(10))

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
#print(X, y)

X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

print(confidence)