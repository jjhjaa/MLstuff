import pandas as pd
import numpy as np
from sklearn.model_selection import LeavePOut
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import zscore

def loadData(): #preparedata
    # LOAD THE DATASET
    df = pd.read_csv("Water_data.csv")
    # APPLY ZSCORE STANDARDIZATION
    df.apply(zscore)
    # EXTRACT VARIABLES
    y = df.iloc[:,3:6].values
    # EXTRACT FEATURES
    X = df.iloc[:,0:3].values
    return X, y, df
# CREATE METHOD FOR CV | LEAVE P OUT CAN BE USED TO LEAVE X AMOUNT OF DATAPOINTS OUT

def LeavePOuts(p_value, X, y, df):
    test_set = []
    predict_set = []
    lpo = LeavePOut(p=p_value)
    # IN THE FOR LOOP WE ALSO FIT THE K-NEARESTNEIGHBOURS REGRESSOR
    for train_index, test_index in lpo.split(df):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knnr = KNeighborsRegressor(n_neighbors=5, metric="minkowski", p=2) # HERE WE ADJUST THE REGRESSOR
        knnr.fit(X_train, y_train) # HERE WE FIT IT
        predictions = knnr.predict(X_test) # TAKE PREDICTIONS
        test_set.append(y_test)
        predict_set.append(predictions)
    print("Cross validation with Leave->",p_value,"<-Out")
    print("KNN Regressor adjustments",knnr)
    print("C-index for c-total:", round(C_index(test_set, predict_set, 0), 5))
    print("C-index for cd:", round(C_index(test_set, predict_set, 1), 5))
    print("C-index for pb:", round(C_index(test_set, predict_set, 2), 5))

# HERE WE CALCULATE THE C INDEX AS WE RECEIVED IT FROM THE PPT SLIDES
def C_index(y_true, predictions, index):
    n = 0
    h_sum = 0
    for i in range(len(y_true)):
        t = y_true[i][0][index]
        p = predictions[i][0][index]
        for j in range(i+1, len(y_true)):
            nt = y_true[j][0][index]
            np = predictions[j][0][index]
            if t != nt:
                n = n+1
                if (p < np and t < nt) or (p > np and t > nt):
                    h_sum += 1
                elif (p < np and t > nt) or (p > np and t < nt):
                    pass
                elif p == np:
                    h_sum += 0.5
    if n != 0:
        return h_sum/n
    return 1
X, y, df = loadData()
LeavePOuts(4, X, y, df) # MAJOR NOTE, HERE YOU SET THE VALUE FOR THE LEAVE-P-OUT METHOD
