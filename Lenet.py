import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

class Lenet:
    """
    Model class with the functions needed for iterative pruning
    """

    network = None
    initial_weights = None  # Keep the very first weights (before training) to be able to reset the network
    prune_percentages = None

    def __init__(self, prune_percentages):
        """
        Args:
            prune_percentages:  Prune percentage for each layer. Example: {'layer0': .2, 'layer1': .2, 'layer2': .1}
        """
        self.network = self.lenet_300_100()
        self.prune_percentages = prune_percentages
        self.initial_weights = self.network.get_weights()  # TODO extract the initial weight from 'network' and store here

        return

    def lenet_300_100(inp=(28,28)):

        model = Sequential()

        model.add(Flatten(input_shape=(28,28)))

        model.add(Dense(300, activation="relu", kernel_initializer="glorot_normal"))
        model.add(Dropout(0.2))

        model.add(Dense(100, activation="relu", kernel_initializer="glorot_normal"))
        model.add(Dropout(0.1))
        model.add(Dense(10, activation="softmax", kernel_initializer="glorot_normal"))

        opt = tf.keras.optimizers.Adam(lr=(1.2*(10**-3)))
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model.compile(optimizer=opt,
                loss=loss_fn,
                metrics=['accuracy'])

        return model

    def reset(self):
        """
        Reset network to original weights.
        """
        self.network.set_weigths(self.initial_weights)

        return

    def get_trainable_weights(self):
        """
        returns the current trainable weights in a dictionary.
        Dict keys are layer names and dict values are the layers in numpy format.
        Could be named like this: {'layer0': w, 'layer1': w, 'layer2': w}
        """
        t_weigths = self.network.get_weights()
        dic_w = {}
        name = "layers"
        for i in range(len(t_weigths)):
            name = name[:-1] + str(i)
            dic_w[name] = t_weigths[i]   

        return dic_w

    def get_prune_percentages(self):
        """
        returns the prune percentages in a dictionary
        """
        return self.prune_percentages

    def train(self, x_train, y_train, x_test,  y_test, masks=None):
        """
        Train the network with the mask if provided.
        May calculate loss and accuracy and store them somewhere for plotting.

        Args:
            masks: A dictionary mapped from layer name -> mask.
            Each mask is in numpy format and with the same dimensions as the weights.
            The keys are the same as for the dectinary keys returned by get_trainable_weights().
        """
        self.network.fit(x_train, y_train, batch_size=60, epochs=5)

        self.network.evaluate(x_test,  y_test, verbose=2)

        return