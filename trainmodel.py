from datetime import time
import pandas as pd
import tqdm
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load the data
df = pd.read_csv("weatherHistory_preprocessed.csv")

# Split the data into training and testing data
# NOTE: In attempt to prevent overfitting, and
# to make the model more accurate, I researched
# The best way to do this and found the following:
# https://medium.com/@tubelwj/five-methods-for-data-splitting-in-machine-learning-27baa50908ed
# This is the resource I used to find this funciton
train_data, test_data = train_test_split(
    df, test_size=0.2, shuffle=True, random_state=42
)

# Initalize the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


# Split the data into features and result
X_train = train_data.drop(columns=["result"])
y_train = train_data["result"]
X_test = test_data.drop(columns=["result"])
y_test = test_data["result"]


# Convert the data to numpy arrays
X_train = X_train.to_numpy(dtype=np.float32)
y_train = y_train.to_numpy(dtype=np.float32)
X_test = X_test.to_numpy(dtype=np.float32)
y_test = y_test.to_numpy(dtype=np.float32)

# Convert the data to tensors
x_train, y_train, x_valid, y_valid = map(
    lambda x: torch.tensor(x).to(device), (X_train, y_train, X_test, y_test)
)


# Print the size of the data
print(f"x_train size: {x_train.shape}")
print(f"y_train size: {y_train.shape}")
print(f"x_valid size: {x_valid.shape}")
print(f"y_valid size: {y_valid.shape}")


# Initalize the model
model = torch.nn.Sequential(
    torch.nn.Linear(39, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 68, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(68, 32, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1, bias=True),
).to(device)


# Print the model
print(model)

# Initialize loss function and optimizer with better parameters
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Initalize the number of epochs and batch size, my attempt at batch size
# Is to try and make the model converge faster and to prevent overfitting
# NOTE: Unsure if the overfitting is happening or not but it does
# Converge much faster
epochs = 200
batch_size = 100

# Initalize best model and best loss
bestModel = None
bestLoss = float("inf")

# Make data to hold information to make graphs
train_loss_data = []
val_loss_data = []
x_val = []

current_train_loss = 0
current_val_loss = 0

# Train the model with X epochs
for epoch in range(epochs):
    # Set up progress bar
    with tqdm.tqdm(total=len(x_train), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
        for i in range(0, len(x_train), batch_size):
            # Get batch of data for this training iteration
            batch_x = x_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]

            # Set model to training mode
            model.train()

            # Forward pass
            y_pred = model(batch_x)

            # Compute training loss
            train_loss = loss_fn(y_pred.squeeze(), batch_y)

            # Backpropagation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Validation phase for this iteration
            model.eval()
            with torch.no_grad():
                # Get predictions for the validation set
                y_val_pred = model(x_valid)
                val_loss = loss_fn(y_val_pred.squeeze(), y_valid)

                # Update best model if current model is better
                # This is the best VALIDATION loss Model
                if val_loss < bestLoss:
                    bestLoss = val_loss
                    bestModel = model.state_dict()

            # Update progress bar
            pbar.set_postfix(
                {
                    "Training Loss": f"{train_loss.item():.3f}",
                    "Validation Loss": f"{val_loss.item():.3f}",
                }
            )  # Show both training and validation loss
            pbar.update(batch_size)  # This is batch size update to the bar

            # Add the data to the graph data
            current_train_loss = train_loss.item()
            current_val_loss = val_loss.item()

    # Add the data to the graph data
    train_loss_data.append(current_train_loss)
    val_loss_data.append(current_val_loss)

    x_val.append(epoch)

# Create the graph
plt.plot(x_val, train_loss_data, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.savefig("training_loss_graph.png")

plt.clf()

# Create the second graph
plt.plot(x_val, val_loss_data, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss")
plt.legend()
plt.savefig("validation_loss_graph.png")

# Load the best model since others may have been worse
model.load_state_dict(bestModel)

# Save the best model
torch.save(model, "models/best_model_torch.torchmodel")

# Final evaluation
model.eval()
with torch.no_grad():
    y_pred = model(x_valid)

    # Calculate several loss metrics
    final_loss = loss_fn(y_pred.squeeze(), y_valid)
    mae = torch.mean(torch.abs(y_pred.squeeze() - y_valid))
    rmse = torch.sqrt(torch.mean((y_pred.squeeze() - y_valid) ** 2))

    # Calculate the average tempature deviation
    avg_temp_dev = torch.mean(torch.abs(y_pred.squeeze() - y_valid))

    # Print the loss metrics
    print(f"Final Test Loss: {final_loss.item():.6f}")
    print(f"MAE: {mae.item():.6f}")
    print(f"RMSE: {rmse.item():.6f}")
    print(f"Average Tempature Deviation: {avg_temp_dev.item():.6f}")

    # Print 5 random predictions and actual values
    for i in range(5):
        random_index = np.random.randint(0, len(x_valid))
        print(
            f"Prediction: {y_pred[random_index].item():.2f}, Actual: {y_valid[random_index].item():.2f}, Diff: {(y_pred[random_index].item() - y_valid[random_index].item()):.2f}"
        )
