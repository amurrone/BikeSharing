import os
import pandas as pd
import argparse

from preprocessing import PreProcessing
from neural_network import NeuralNetwork

from sklearn.model_selection import train_test_split

csv_hourly = "hour.csv"
csv_daily = "day.csv"


# List of input features divided into categorical and numerical features
# Instant and dteday are not used.
# Casual and registered users are of course not used when building the predictive model.
categorical_features = ["yr", "season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
numerical_features = ["atemp", "temp", "hum", "windspeed", "cnt"]
target = ["cnt"]

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", help="path to the input directory")
args = parser.parse_args()

bike_sharing_path = args.dir

if __name__ == "__main__":

    data_hourly = PreProcessing.load_data(bike_sharing_path, csv_hourly)
    data_daily = PreProcessing.load_data(bike_sharing_path, csv_daily)


    # =================
    # Data description
    # =================

    # dteday: date
    # season: season (1:winter, 2:spring, 3:summer, 4:fall)
    # yr: year (0: 2011, 1:2012)
    # mnth: month (1 to 12)
    # hr: hour (0 to 23)
    # holiday: (0:no, 1:yes)
    # weekday: day of the week
    # workingday: (0:no, 1:yes)
    # weathersit:
        #- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
        #- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
        #- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
        #- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
    # temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
    # atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
    # hum: Normalized humidity. The values are divided to 100 (max)
    # windspeed: Normalized wind speed. The values are divided to 67 (max)
    # casual: count of casual users
    # registered: count of registered users
    # cnt: count of total rental bikes including both casual and registered

    #print (data_hourly.head())
    #print (data_hourly.describe())


    # One hot encoding of categorical features

    list_final_features = []
    list_labels = []

    for cat in categorical_features:
        feature_ohe = PreProcessing.fit_transform_ohe(data_hourly, cat)
        list_final_features.append(feature_ohe)

    for num in numerical_features:
        list_final_features.append(data_hourly[num])
        list_labels.append(num)

    new_data_hourly = pd.concat(list_final_features, axis=1)
    new_features = (new_data_hourly.columns).tolist()

    new_features.remove("cnt")


    # Remove outliers in cnt (count rate)
    new_data_hourly = PreProcessing.remove_outliers(new_data_hourly, "cnt", 3)

    # Split the dataset into train and test samples
    train_set, test_set = train_test_split(new_data_hourly, test_size=0.33, random_state=123)

    feats_train = train_set.reindex(columns=new_features).values
    feats_test = test_set.reindex(columns=new_features).values

    target_train = train_set.reindex(columns=target).values.ravel()
    target_test = test_set.reindex(columns=target).values.ravel()

    #train_set.hist(column="cnt", bins=1000)
    #test_set.hist(column="cnt", bins=1000)

    # Build the neural network
    dnn = NeuralNetwork(100, 3)
    dnn_model = dnn.build_model(activation="relu", batch_norm="True")

    # Train the neural network
    dnn.train(model=dnn_model, optimizer="adam", loss="mse", metrics="accuracy", features=feats_train, target=target_train, epochs=10, verbose=1, validation_split=0.2, trained_model="bike_sharing_trained")


    #print (dnn_model.summary())
