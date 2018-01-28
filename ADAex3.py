import pandas as pd
from sklearn.model_selection import LeavePGroupsOut
from sklearn.neighbors import KNeighborsClassifier

# Setters
zscore = lambda x: (x - x.mean()) / x.std()

#----------------------------------- DATA PREPROCESSING -----------------------------------
# Load data
df = pd.read_csv("painsignals.csv")
# Extract features
X = ["hr", "rrpm", "gsr", "rmscorr", "rmsorb"]
# Extract variable
y = df[["label"]].values
# Extract subject groups
sg = df.iloc[:,0].values
# apply zscore standardization
dfs = df.copy()
dfs[X] = \
    dfs[X].groupby(dfs.subject).transform(zscore)
X = dfs[X].values

# ----------------------------------- C INDEX -----------------------------------
# C_index algo from lecture slides
def C_index(y_true, predictions):
    n = 0
    h_sum = 0
    for i in range(len(y_true)):
        t = y_true[i]
        p = predictions[i]
        for j in range(i+1, len(y_true)):
            nt = y_true[j]
            np = predictions[j]
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

# ----------------------------------- CROSS VALIDATION & CLASSIFICATION -----------------------------------
C_score = []
lpgo = LeavePGroupsOut(n_groups=1) # Set LPGO
for train, test in lpgo.split(X, y, sg):
    X_train, X_test = X[train], X[test] # Divide features
    y_train, y_test = y[train], y[test] # Divide variable
    knnc = KNeighborsClassifier(n_neighbors=37, n_jobs=-1) # Adjust the classifier
    knnc.fit(X_train, y_train.ravel()) # Fit classificator
    predictions = knnc.predict(X_test) # Take out predictions
    C_instance = C_index(y_test, predictions) # Calculate one instance
    C_score.append(C_instance) # add iterations to C score

    # ----------------------------------- SHOW SCORES -----------------------------------
    print("Iteration", len(C_score), "Concordance index:", C_instance)
avg = sum(C_score)/float(len(C_score))
print("The average index of C instances:", avg)