# ==============================================================================
# Class dedicated to the preprocessing of data
# ==============================================================================

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class PreProcessing():

    def load_data(path, csv):
        """
        Load data

        Parameters:
        ===========
        path: path to the file
        csv: name of the csv file

        Return:
        =======
        Pandas dataframe with the loaded data
        """
        csv_file = os.path.join(path, csv)
        return pd.read_csv(csv_file)


    # Encording of categorical features

    def fit_transform_ohe(df, col):
        """
        One hot encoding for the specified column

        Parameters:
        ===========
        df: pandas.DataFrame containing the specified column
        col: the specific column on which performing one hot encoding

        Return:
        =======
        feature_df: new pandas.Dataframe with the specified column one hot encoded
        """

        le = LabelEncoder()
        ohe = OneHotEncoder()
        le_labels = le.fit_transform(df[col])
        feature_arr = ohe.fit_transform(df[[col]]).toarray()
        feature_labels = [col+"_"+str(label) for label in le.classes_]
        feature_df = pd.DataFrame(feature_arr, columns=feature_labels)

        return feature_df

    # Remove outliers

    def remove_outliers(df, col, std):
        """
        Remove outliers

        Parameters:
        ===========
        df: pandas.DataFrame containing the specified column
        col: column (input feature)
        std: how many standard deviations should the value be from the mean value to be considered as outlier (3 is a standard convention)

        Return:
        =======
        new_df: new pandas.Dataframe with outliers removed
        """
        new_df = df[np.abs(df[col] - df[col].mean()) <= std*df[col].std()]

        return new_df
