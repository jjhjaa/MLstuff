import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
#import file
df = pd.read_csv('Concrete_Data.csv')

#independent variables
X = df.iloc[: ,:-1].values
#depent variable
y = df.iloc[:, 8].values
#print(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

'''
#Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
'''

#Classifier fitting
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

confidence = regressor.score(X_test, y_test)
print(confidence)
#0.636960651834
#plt.scatter(X_train, y_train, color = 'red')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#plt.title('cement')