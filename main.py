import os
import pandas as pd

from preprocessing import PreProcessing

bike_sharing_path = "/Users/Alessia/Desktop/Bike_Sharing/Bike-Sharing-Dataset/"
csv_hourly = "hour.csv"
csv_daily = "day.csv"

categorical_features = ["yr", "season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
numerical_features = ["atemp", "temp", "hum", "windspeed", "cnt"]

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
    # Features are listed in categorical_features and numerical_features lists.
    # Instant and dteday are not used.
    # Casual and registered users are of course not used when building the predictive model.

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

    #print (new_data_hourly)

    # Remove outliers in "cnt" (count rate)
    new_data_hourly = PreProcessing.remove_outliers(new_data_hourly, "cnt", 3)

    #print (new_data_hourly)
