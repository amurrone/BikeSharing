import os
import pandas as pd
import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt

from preprocessing import PreProcessing
from neural_network import NeuralNetwork

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

logging.basicConfig(level=logging.INFO)


def plot_loss(history):
    """
    Generate loss vs epoch plot.

    Parameters
    ==========
    history : history attribute of History object (History.history).
    According to tensorflow.keras this a record of
    training (validation) loss values and training (validation) metrics at successive epochs
    """

    fig = plt.figure()
    plt.plot(history['loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    return fig

def plot_prediction(true, predicted):
    """
    Generate true vs predicted plot.

    Parameters
    ==========
    true: np.array with true values
    predicted: np.array with values predicted by the algorithm
    """

    fig = plt.figure()
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.plot(true, predicted, "b.", markersize=9)
    true_max = true.max()
    true_min = true.min()
    predicted_max = predicted.max()
    predicted_min = predicted.min()
    lineEnd = true_max if true_max > predicted_max else predicted_max
    lineStart = true_min if true_min < predicted_min else predicted_min
    plt.plot([lineStart-10, lineEnd+10], [lineStart-10, lineEnd+10], 'k-', color = 'r')
    plt.xlim(lineStart-10, lineEnd+10)
    plt.ylim(lineStart-10, lineEnd+10)
    return fig



if __name__ == "__main__":

    logging.info("Loading data in {}".format(bike_sharing_path))

    data_hourly = PreProcessing.load_data(bike_sharing_path, csv_hourly)
    data_daily = PreProcessing.load_data(bike_sharing_path, csv_daily)

    logging.info("Data succsesfully loaded!")


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
    logging.info("Performing one hot encoding for categorical features")

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

    # Remove target feature from the inputs
    new_features.remove("cnt")

    # Remove atemp feature as this is highly correlated with temp (see correlation map) and we want to avoid multicollinearity problems in the neural network
    new_features.remove("atemp")


    # Remove outliers in cnt (count rate)
    new_data_hourly = PreProcessing.remove_outliers(new_data_hourly, "cnt", 3)


    # Split the dataset into train and test samples
    train_set, test_set = train_test_split(new_data_hourly, test_size=0.33, random_state=123)

    feats_train = train_set.reindex(columns=new_features).values
    feats_test = test_set.reindex(columns=new_features).values

    target_train = train_set.reindex(columns=target).values.ravel()
    target_test = test_set.reindex(columns=target).values.ravel()

    # Check wether there is some bias in train and test samples
    train_set.hist(column="cnt", bins=1000)
    plt.savefig('plots/NeuralNetwork/train_histo.png')
    test_set.hist(column="cnt", bins=1000)
    plt.savefig('plots/NeuralNetwork/test_histo.png')


    # Build the neural network
    dnn = NeuralNetwork(neurons=100, hidden_layers=3)
    dnn_model = dnn.build_model(activation="relu", batch_norm="True", optimizer="adam", loss="mse", metrics="accuracy")

    # Train the neural network
    logging.info("Training the neural network")
    dnn_training = dnn.train(model=dnn_model, features=feats_train, target=target_train, epochs=10, verbose=1, validation_split=0.2, trained_model="bike_sharing_trained")
    history = dnn_training[0]

    print (dnn_model.summary())

    # Evaluate the training performances
    dnn_prediction_train = dnn.test(trained_model="bike_sharing_trained", features=feats_train)

    dnn_mae_train = mean_absolute_error(dnn_prediction_train, target_train)
    dnn_mse_train = mean_squared_error(dnn_prediction_train, target_train)
    dnn_rmse_train = np.sqrt(dnn_mse_train)


    print ("True value training:", target_train)
    print ("DNN prediction training:", dnn_prediction_train)

    print ("Mean absolute error training:", dnn_mae_train)
    print ("Root mean squared error training:", dnn_rmse_train)


    # Plot loss function vs epoch
    logging.info("Plotting loss function vs epoch")
    loss_curve = plot_loss(history.history)
    loss_curve.savefig('plots/NeuralNetwork/loss.png')


    # Test the neural network
    logging.info("Testing the neural network")
    dnn_prediction_test = dnn.test(trained_model="bike_sharing_trained", features=feats_test)

    dnn_mae_test = mean_absolute_error(dnn_prediction_test, target_test)
    dnn_mse_test = mean_squared_error(dnn_prediction_test, target_test)
    dnn_rmse_test= np.sqrt(dnn_mse_test)

    print ("True value testing:", target_test)
    print ("DNN prediction testing:", dnn_prediction_test)

    print ("Mean absolute error testing:", dnn_mae_test)
    print ("Root mean squared error testing:", dnn_rmse_test)

    # Plot true vs predicted values
    true_vs_predicted = plot_prediction(target_test, dnn_prediction_test)
    true_vs_predicted.show()
    true_vs_predicted.savefig('plots/NeuralNetwork/true_vs_predicted.png')
