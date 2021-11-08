# ========================================
# Class dedicated to preprocessing of data
# ========================================

import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class PreProcessing():

    def load_data(path, csv):
        """
        Load data.

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
