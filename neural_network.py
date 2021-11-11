# ===================================
# Neural network class
# ===================================

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
tf.random.set_seed(0)
from sklearn.metrics import mean_squared_error, mean_absolute_error

class NeuralNetwork():

    def __init__(self, neurons: int, hidden_layers: int):
        """
        Parameters:
        ===========
        neurons: number of neurons of each layer. The number is supposed to be the same for each layer.
        hidden_layers: number of hidden layers.
        """
        self.neurons = neurons
        self.hidden_layers = hidden_layers

    def build_model(self, activation, batch_norm):
        """
        Build the neural network regression model

        Parameters:
        ===========
        activation: activation function, same for all the hidden layers. No activation function in the last layer since this is a regression model and we want directly the predicted output.
        batch_norm: apply batch normalization algorithm.

        Return:
        =======
        Neural network model.
        """

        model = Sequential()
        for h in range(self.hidden_layers):
            model.add(BatchNormalization()) if batch_norm else None
            model.add(Dense(self.neurons, activation))
        model.add(Dense(1))

        return model
