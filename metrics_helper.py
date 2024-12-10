# This file is designed to take in 2 numpy arrays and calculate the evaluation metrics
# It will then save the evaluation metrics to a text file
# NOTE: The original implmentation of my metrics used forumlas from here:# Calculate differnt error values/evaluation metrics
# SOURCE: https://medium.com/analytics-vidhya/complete-guide-to-machine-learning-evaluation-metrics-615c2864d916
# I used this source to get the forumlas and to better understand the evaluation metrics
# And what they tell me about the model I created
# I then found that the metrics can be calculated using sklearn
# So i updated the code to use sklearn
# SOURCE: https://scikit-learn.org/stable/modules/model_evaluation.html

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_evaluation_metrics(predictions, results, name):
    # Calculate the evaluation metrics
    mse = mean_squared_error(results, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(results, predictions)
    avgTempDiff = np.mean(np.abs(predictions - results))

    # Save the evaluation metrics to a text file
    with open(f"metrics/evaluation_metrics_{name}.txt", "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"Root Mean Squared Error: {rmse}\n")
        f.write(f"R2: {r2}\n")
        f.write(f"Average Temperature Difference: {avgTempDiff}\n")

    # Print the evaluation metrics
    print("----------------------------------------")
    print(f"|Model: {name}")
    print(f"|Mean Squared Error: {mse}")
    print(f"|Root Mean Squared Error: {rmse}")
    print(f"|R2: {r2}")
    print(f"|Average Temperature Difference: {avgTempDiff}")
    print("----------------------------------------")


def average_temp_diff(predictions, results):
    return np.mean(np.abs(predictions - results))
