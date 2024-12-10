# Get all the files evaulation_metrics_*.txt and apply processing
# To generate information for the report
import os
from matplotlib import pyplot as plt
import prettytable as pt


class EvaluationMetrics:
    def __init__(self, lines):
        # Split the lines by the colon and get the second element(Which is the value)
        self.model_name = lines[0].split(": ")[1].strip()
        self.mae = float(lines[1].split(": ")[1].strip())
        self.mse = float(lines[2].split(": ")[1].strip())
        self.rmse = float(lines[3].split(": ")[1].strip())
        self.r2 = float(lines[4].split(": ")[1].strip())
        self.avg_temp_diff = float(lines[5].split(": ")[1].strip())

    def __str__(self):
        return f"Model: {self.model_name}\nMAE: {self.mae}\nMSE: {self.mse}\nRMSE: {self.rmse}\nR2: {self.r2} Avg Temp Diff: {self.avg_temp_diff}"

    def __repr__(self):
        return f"Model: {self.model_name}\nMAE: {self.mae}\nMSE: {self.mse}\nRMSE: {self.rmse}\nR2: {self.r2} Avg Temp Diff: {self.avg_temp_diff}"

    # For sorting
    def __gt__(self, other):
        # Check if the avg temp diff is greater than the other
        return self.avg_temp_diff > other.avg_temp_diff

    def __lt__(self, other):
        # Check if the avg temp diff is less than the other
        return self.avg_temp_diff < other.avg_temp_diff

    def __eq__(self, other):
        # Check if the avg temp diff is equal to the other
        return self.avg_temp_diff == other.avg_temp_diff

    # For making nice tables
    def get_name(self):
        return self.model_name

    # Round the values to 5 decimal places
    def get_mae(self):
        return round(self.mae, 5)

    def get_mse(self):
        return round(self.mse, 5)

    def get_rmse(self):
        return round(self.rmse, 5)

    def get_r2(self):
        return round(self.r2, 5)

    def get_avg_temp_diff(self):
        return round(self.avg_temp_diff, 5)


# List to store all the metrics
metrics = []
# Fill in the metrics list
for f in os.walk("metrics"):
    for file in f[2]:
        file = os.path.join("metrics", file)
        if file.startswith("metrics/evaluation_metrics_") and file.endswith(".txt"):
            print(f"Found File that matches pattern: {file}")
            with open(file, "r") as file:
                lines = file.readlines()
                metrics.append(EvaluationMetrics(lines))


# Sort the metrics by the average temperature difference
metrics.sort()

# Create a table to display the metrics
table = pt.PrettyTable()
table.field_names = ["Model Name", "MSE", "RMSE", "R2", "Mean Temp Diff (C)"]
for metric in metrics:
    table.add_row(
        [
            metric.get_name(),
            metric.get_mse(),
            metric.get_rmse(),
            metric.get_r2(),
            metric.get_avg_temp_diff(),
        ]
    )

print(table)
