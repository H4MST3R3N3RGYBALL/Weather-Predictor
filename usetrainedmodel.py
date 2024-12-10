import torch
import pandas as pd
import numpy as np
from metrics_helper import calculate_evaluation_metrics


# Make a class to load the best model and use it to make predictions
class BestModel:
    def __init__(self, model_path):
        self.model = torch.load(model_path, weights_only=False)
        # Get the device including mps
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        print(f"Using {self.device} device")

        # Move the model to the device
        self.model.to(self.device)

    def predict(self, x):
        # Move the data to the device
        x = x.to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(x).cpu()

    def predict_all(self, x):
        # Move the data to the device
        x = x.to(self.device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(len(x)):
                pred = self.model(x[i]).cpu().item()
                predictions.append(pred)
                if i % 1000 == 0:
                    print(f"Predicted {i} out of {len(x)}")
        return predictions


# Create an instance of the BestModel class
best_model = BestModel("models/best_model.torchmodel")

# Load weather history Processed Data
weather_history_base = pd.read_csv("weatherHistory_preprocessed.csv")

# Drop the result column
weather_history = weather_history_base.drop(columns=["result"])

# Convert the data to numpy arrays
weather_history = weather_history.to_numpy(dtype=np.float32)

# Convert the numpy array to a tensor
weather_history = torch.tensor(weather_history)

# Make predictions (The class puts them on the cpu)
predictions = best_model.predict_all(weather_history)

# Convert Predictions to a numpy array
predictions = np.array(predictions, dtype=np.float32)
res = weather_history_base["result"].to_numpy(dtype=np.float32)

# Round the predictions and results to 2 decimal places
predictions = np.round(predictions, 2)
res = np.round(res, 2)


# Save the predictions and results to a csv file
np.set_printoptions(suppress=True)
np.savetxt("predictions_nn.csv", predictions, delimiter="\n", fmt="%.2f")
np.savetxt("results_nn.csv", res, delimiter="\n", fmt="%.2f")

# Calculate the evaluation metrics
calculate_evaluation_metrics(predictions, res, "nn")
