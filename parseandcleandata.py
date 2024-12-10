import pandas as pd

# Load the data into a pandas dataframe
df = pd.read_csv("weatherHistory.csv")
# print(df.head())

# Print all unique values of summary and precip type
# print(df['Summary'].unique())
# print(df['Precip Type'].unique())

# All of the following is the preprocessing of the data

# Drop the columns that are not needed
df = df.drop(columns=["Daily Summary"])

# Convert the Formatted Date to a datetime object
df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)

df["month"] = df["Formatted Date"].dt.month
df["day"] = df["Formatted Date"].dt.day
df["hour"] = df["Formatted Date"].dt.hour


# Drop the Formatted Date column now that we have extracted
df = df.drop(columns=["Formatted Date"])


# For the categorical data, we will use one-hot encoding
# I found this by looking up how to turn categorical data into numerical data
# https://www.geeksforgeeks.org/ml-one-hot-encoding/
df = pd.get_dummies(df, columns=["Summary", "Precip Type"])

# Rename the Temperature (C) column to result
df = df.rename(
    columns={"Temperature (C)": "result"}
)  # This is the column we are trying to predict
# Rename apparent temperature (C) to apptemp
df = df.rename(columns={"Apparent Temperature (C)": "apptemp"})
# Rename wind speed (km/h) to windspeed
df = df.rename(columns={"Wind Speed (km/h)": "windspeed"})
# Rename Wind Bearing (degrees) to windbearing
df = df.rename(columns={"Wind Bearing (degrees)": "windbearing"})
# Rename Visibility (km) to visibility
df = df.rename(columns={"Visibility (km)": "visibility"})
# Rename Loud Cover to loudcover
df = df.rename(columns={"Loud Cover": "loudcover"})
# Rename Pressure (millibars) to pressure
df = df.rename(columns={"Pressure (millibars)": "pressure"})
# Rename Humidity to humidity(just for caps consistency)
df = df.rename(columns={"Humidity": "humidity"})


# Apply the Fill the missing values with 0
df = df.fillna(0)


# Normalize the int and float columns to be between 0 and 1 to all but the result column
# Apptemp, windspeed, windbearing, visibility, loudcover, pressure, humidity
# This is done by shifting by the minimum and dividing by the range
# NOTE FOR REPORT: THIS WAS CAUSING HUGE ERRORS IN THE MODEL UNSURE WHY
# MAYBE EXPLORE THIS LATER
# df['apptemp'] = (df['apptemp'] - df['apptemp'].min()) / (df['apptemp'].max() - df['apptemp'].min())
# df['windspeed'] = (df['windspeed'] - df['windspeed'].min()) / (df['windspeed'].max() - df['windspeed'].min())
# df['windbearing'] = (df['windbearing'] - df['windbearing'].min()) / (df['windbearing'].max() - df['windbearing'].min())
# df['visibility'] = (df['visibility'] - df['visibility'].min()) / (df['visibility'].max() - df['visibility'].min())
# df['loudcover'] = (df['loudcover'] - df['loudcover'].min()) / (df['loudcover'].max() - df['loudcover'].min())
# df['pressure'] = (df['pressure'] - df['pressure'].min()) / (df['pressure'].max() - df['pressure'].min())
# df['humidity'] = (df['humidity'] - df['humidity'].min()) / (df['humidity'].max() - df['humidity'].min())


print(df.head())

# Save the preprocessed data to a new csv file
df.to_csv("weatherHistory_preprocessed.csv", index=False)
