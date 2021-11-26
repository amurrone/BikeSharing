# ==============================================================================
# Neural network class
# ==============================================================================

import datetime

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import load_model
tf.random.set_seed(0)
from sklearn.metrics import mean_squared_error, mean_absolute_error


class NeuralNetwork():

    def __init__(self, neurons: int, hidden_layers):
        """
        Parameters:
        ===========
        neurons: number of neurons of each layer. The number is supposed to be the same for each layer.
        hidden_layers: number of hidden layers
        """

        self.neurons = neurons
        self.hidden_layers = hidden_layers

    def build_model(self, activation, batch_norm, optimizer, loss, metrics):
        """
        Build the neural network regression model

        Parameters:
        ===========
        activation: activation function, same for all the hidden layers. No activation function in the last layer since this is a regression model and we want directly the predicted output.
        batch_norm: apply batch normalization algorithm
        optimizer: optimizer algorithm
        loss: loss function
        metrics: metrics

        Return:
        =======
        tensorflow.keras neural network model
        """

        model = Sequential()
        for h in range(self.hidden_layers):
            model.add(BatchNormalization()) if batch_norm else None
            model.add(Dense(self.neurons, activation=activation))
        model.add(Dense(1))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

    def train(self, model, features, target, epochs, verbose, validation_split, trained_model):
        """
        Train the neural network regression model

        Parameters:
        ===========
        model: tf.keras neural network model
        features: input features
        target: target features
        epochs: number of epochs
        verbose: verbose
        validation_split: fraction of training dataset allocated for validation
        trained_model: name of the train model to be saved

        Return:
        =======
        tensorflow.keras history object
        tensorflow.keras trained model
        """

        log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(features, target, epochs=epochs, verbose=verbose, validation_split=validation_split, callbacks=[tensorboard_callback])
        model.save("saved_models/"+trained_model+".h5")

        return history, model

    def test(self, trained_model, features):
        """
        Test the neural network regression model

        Parameters:
        ===========
        trained_model: name of the train model saved
        features: input features

        Return:
        =======
        tensorflow.keras model prediction
        """

        model = tf.keras.models.load_model("saved_models/"+trained_model+".h5")
        prediction = model.predict(features)

        return prediction
