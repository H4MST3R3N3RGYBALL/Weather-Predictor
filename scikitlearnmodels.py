from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    RANSACRegressor,
    HuberRegressor,
    ElasticNet,
    SGDRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from metrics_helper import calculate_evaluation_metrics, average_temp_diff


import joblib

# Load the data
data = pd.read_csv("weatherHistory_preprocessed.csv")

# Split the data into training and testing data
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)


# Split the data into features and result and convert to numpy arrays
X_train = train_data.drop(columns=["result"]).to_numpy()
y_train = train_data["result"].to_numpy()
X_test = test_data.drop(columns=["result"]).to_numpy()
y_test = test_data["result"].to_numpy()


bestAverageTempDiff = float("inf")
bestModel = ""

# Try using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "linear-regression")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
    bestModel = "linear-regression"
joblib.dump(model, "models/linear-regression-model.pkl", compress=9)

# Try using Lasso
model = Lasso()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "lasso")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
    bestModel = "lasso"
joblib.dump(model, "models/lasso-model.pkl", compress=9)


# Try using Ridge
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "ridge")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
    bestModel = "ridge"
joblib.dump(model, "models/ridge-model.pkl", compress=9)

# Try using RANSACRegressor
model = RANSACRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "ransac-regressor")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
    bestModel = "ransac-regressor"
joblib.dump(model, "models/ransac-regressor-model.pkl", compress=9)


# Try using HuberRegressor
model = HuberRegressor(max_iter=100000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "huber-regressor")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
    bestModel = "huber-regressor"
joblib.dump(model, "models/huber-regressor-model.pkl", compress=9)

# Try using ElasticNet
model = ElasticNet(max_iter=100000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "elasticnet")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
    bestModel = "elasticnet"
joblib.dump(model, "models/elasticnet-model.pkl", compress=9)


# Try using SGDRegressor
# SGD has some variance, so we will run it 10 times and take the best result
localBest = float("inf")
localBestModel = None
localMdl = None
for x in range(10):
    model = SGDRegressor(
        shuffle=True,
        penalty="elasticnet",
        alpha=0.0001,
        loss="huber",
        max_iter=1000000,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    avgTempDiff = average_temp_diff(predictions, y_test)
    if avgTempDiff < localBest:
        localBest = avgTempDiff
        localBestModel = predictions
        localMdl = model
calculate_evaluation_metrics(localBestModel, y_test, "sgd-regressor")
if localBest < bestAverageTempDiff:
    bestAverageTempDiff = localBest
    bestModel = "sgd-regressor"
joblib.dump(localMdl, "models/sgd-regressor-model.pkl", compress=9)

# Try using SVR with RBF kernel
model = SVR(verbose=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "svr-rbf")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
    bestModel = "svr-rbf"
joblib.dump(model, "models/svr-rbf-model.pkl", compress=9)

# Try using poly
model = SVR(kernel="poly", verbose=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "svr-poly")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
    bestModel = "svr-poly"
joblib.dump(model, "models/svr-poly-model.pkl", compress=9)

# Try using sigmoid
model = SVR(kernel="sigmoid", verbose=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "svr-sigmoid")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
bestModel = "svr-sigmoid"
joblib.dump(model, "models/svr-sigmoid-model.pkl", compress=9)

# Try using Decision Tree Regressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "decision-tree")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
    bestModel = "decision-tree"
joblib.dump(model, "models/decision-tree-model.pkl", compress=9)

# Try using Random Forest Regressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "random-forest")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
    bestModel = "random-forest"
joblib.dump(model, "models/random-forest-model.pkl", compress=9)

# Try using Gradient Boosting Regressor
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "gradient-boosting")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
    bestModel = "gradient-boosting"
joblib.dump(model, "models/gradient-boosting-model.pkl", compress=9)

# Try using KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "k-neighbors")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
    bestModel = "k-neighbors"
joblib.dump(model, "models/k-neighbors-model.pkl", compress=9)

# Try using MLPRegressor
model = MLPRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "mlp-regressor")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
    bestModel = "mlp-regressor"
joblib.dump(model, "models/mlp-regressor-model.pkl", compress=9)

# Try using AdaBoostRegressor
model = AdaBoostRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
calculate_evaluation_metrics(predictions, y_test, "adaboost")
avgTempDiff = average_temp_diff(predictions, y_test)
if avgTempDiff < bestAverageTempDiff:
    bestAverageTempDiff = avgTempDiff
    bestModel = "adaboost"
joblib.dump(model, "models/adaboost-model.pkl", compress=9)


print(f"Best model: {bestModel}")
print(f"Best average temperature difference: {bestAverageTempDiff}")
