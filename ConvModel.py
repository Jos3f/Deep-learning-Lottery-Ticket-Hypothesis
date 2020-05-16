from tensorflow.keras.datasets import cifar10
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import MaxPooling2D, Conv2D
from tensorflow.keras import optimizers

import tensorflow_model_optimization as tfmot


class ConvModel:
    """
    Conv-2,4 and 6 model class with the functions needed for iterative pruning
    """

    input_data_shape = 1

    def __init__(self, input_data_shape, model_name='conv-6', use_dropout=False):

        # Hyper-params that can be changed
        self.K = 10
        self.learning_rate = 0.0003
        self.batch_size = 60

        self.input_data_shape = input_data_shape
        self._use_dropout = use_dropout

        assert model_name in ['conv-2', 'conv-4', 'conv-6'], "Use Conv-2, Conv-4 or Conv-6 model"
        self._model_name = model_name

        # All the layers with relevant weights for the pruning
        self._weight_layers = []

        # The percentages we should prune for each layer. Eg. { "conv-1": 0.2 ... }
        self._prune_percentages = {}

        # Initially build it with no pruning
        self._build()
        self._compile()

        # The initial weights are stored the first time we build the model only,
        # so we can reset it when we change the pruning
        self._initial_weights = self._get_trainable_weights()

        # Save last used masks so that it can be used when we reinitialize the layers
        self._initial_random_weights = None
        self._last_used_masks = None

    def _build(self):
        """
        Builds the model from scratch. If we have set the pruning percentages we will build the model with pruning.
        """

        self._weight_layers = []
        self._model = Sequential()

        self._add_layer(Conv2D(64, (3, 3), name='conv-1', padding='same', activation='relu', input_shape=self.input_data_shape, kernel_initializer='glorot_normal'))
        self._add_layer(Conv2D(64, (3, 3), name='conv-2', padding='same', activation='relu', kernel_initializer='glorot_normal'))
        self._add_layer(MaxPooling2D((2, 2)))
        self._weight_layers.extend(['conv-1', 'conv-2'])

        if self._model_name in ['conv-2', 'conv-4']:
            self._add_layer(Conv2D(128, (3, 3), name='conv-3', padding='same', activation='relu', kernel_initializer='glorot_normal'))
            self._add_layer(Conv2D(128, (3, 3), name='conv-4',  padding='same', activation='relu', kernel_initializer='glorot_normal'))
            self._add_layer(MaxPooling2D((2, 2)))
            self._weight_layers.extend(['conv-3', 'conv-4'])

        if self._model_name == 'conv-6':
            self._add_layer(Conv2D(256, (3, 3), name='conv-5', padding='same', activation='relu', kernel_initializer='glorot_normal'))
            self._add_layer(Conv2D(256, (3, 3), name='conv-6', padding='same', activation='relu', kernel_initializer='glorot_normal'))
            self._add_layer(MaxPooling2D((2, 2)))
            self._weight_layers.extend(['conv-5', 'conv-6'])

        self._add_layer(Flatten())
        if self._use_dropout: self._add_layer(Dropout(0.5))
        self._add_layer(Dense(256, name='dense-1', activation='relu', kernel_initializer='glorot_normal'))
        if self._use_dropout: self._add_layer(Dropout(0.5))
        self._add_layer(Dense(256, name='dense-2', activation='relu', kernel_initializer='glorot_normal'))
        if self._use_dropout: self._add_layer(Dropout(0.5))
        self._add_layer(Dense(self.K, name='dense-3', activation='softmax', kernel_initializer='glorot_normal'))
        self._weight_layers.extend(['dense-1', 'dense-2', 'dense-3'])


    def _add_layer(self, layer):
        """
        Helper for adding layer to the current model, also sets the pruning if we have a pruning percentage for the layer
        :param layer: The layer we should add
        """
        if self._prune_percentages is not None and layer.name in self._prune_percentages:
            return self._model.add(
                tfmot.sparsity.keras.prune_low_magnitude(
                    layer,
                    pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(self._prune_percentages[layer.name],
                                                                           begin_step=0, end_step=1, frequency=1),
                    name=layer.name
                )
            )
        self._model.add(layer)



    def _compile(self):
        adam = optimizers.Adam(self.learning_rate)
        self._model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    def predict(self, x):
        """
        Make predictions
        :param x: input data
        :return: output
        """
        return self._model.predict(x)

    def evaluate(self, x, y):
        """
        Evaluate a model on test data
        :param x: input
        :param y: labels
        """
        eval = self._model.evaluate(x, y)
        print("Accuracy: " + str(eval[1]) + ", Loss: " + str(eval[0]))
        return eval

    def train(self, x_train, y_train, iterations=30000, early_stopping=False):
        """
        Train the network. If we want pruning the method set pruning should be called prior to this
        :param x_train: training inputs
        :param y_train: training labels
        :param iterations: number of iterations to run, will be rounded to nearest epoch count
        :param early_stopping: True if early stopping should be used
        """

        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        val_split = 0.0
        if early_stopping:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
            val_split = 0.2
            callbacks.append(early_stopping)

        epochs = int(iterations * self.batch_size / (x_train.shape[0] * (1 - val_split) ))
        return self._model.fit(x_train, y_train, self.batch_size, epochs, callbacks=callbacks, validation_split=val_split)



    def set_pruning(self, prune_percentages=None):
        """
        This method resets the network, applies pruning that will be used on next training round and re-initializes
        the unpruned weights to their initial values.
        :param prune_percentages: Dictionary of pruning values to use. Eg. { 'conv-1': 0.2, ... }
        """

        # Create a mask based on the current values in the network
        masks = {}

        # Prunes the additional smallest weights apart from the ones in earlier steps. Uses the weights
        # from the final step in the earlier training session
        for name, weights in self._get_trainable_weights().items():
            # Calculate how many weights we need to prune
            prune_count = int(round(prune_percentages[name] * weights.size))

            # Sort by the magnitude of the weights for the weights and extract the element value where
            # we have gone past 'prune_count' nodes.
            cutoff_value = np.sort(np.absolute(weights.ravel()))[prune_count]

            # Update mask by disabling the nodes with magnitude smaller than/equal to our cutoff value
            masks[name] = np.where(np.absolute(weights) > cutoff_value, 1, 0)

        # Store last used mask globally
        self._last_used_masks = masks

        # Set new prune percentages and rebuild
        self._prune_percentages = prune_percentages
        self._build()
        self._compile()

        # Set initial weights and mask already pruned values.
        # So when we start the training it will prune away the values that was pruned in the previous training round
        # plus (p - p_previous) % of the lowest values where p is the current pruning amount and p_previous was the
        # pruning percentage from the previous training session
        for name in self._weight_layers:
            layer = self._get_layer(name)
            weights = layer.get_weights()
            weights[0] = self._initial_weights[name] * masks[name]
            layer.set_weights(weights)

    def set_pruning_random_init(self, prune_percentages=None):
        """
        This method resets the network, applies pruning that will be used on next training round and re-initializes
        the unpruned weights to their initial values.
        :param prune_percentages: Dictionary of pruning values to use. Eg. { 'conv-1': 0.2, ... }
        """

        # Use the most recent masks
        masks = self._last_used_masks


        # Set new prune percentages and rebuild
        self._prune_percentages = prune_percentages
        self._build()
        self._compile()

        # Use the random weights
        if self._initial_random_weights is None:
            self._initial_random_weights = self._get_trainable_weights()
            return

        # Set weights and mask already pruned values.
        # So when we start the training it will prune away the values that was pruned in the previous training round
        # plus (p - p_previous) % of the lowest values where p is the current pruning amount and p_previous was the
        # pruning percentage from the previous training session
        for name in self._weight_layers:
            layer = self._get_layer(name)
            weights = layer.get_weights()
            weights[0] = self._initial_random_weights[name] * masks[name]
            layer.set_weights(weights)

    def checkpoint_weights(self):
        """Create a check point so that we do not lose trained weights if needed for pruning"""
        self.last_used_trainable = self._get_trainable_weights()


    def reset_to_old_weights(self):
        """Reset the model to the weights that are in our check point"""
        for name, weights_m in self.last_used_trainable.items():
            layer = self._get_layer(name)
            weights = layer.get_weights()
            weights[0] = weights_m
            layer.set_weights(weights)
        pass


    def _get_trainable_weights(self):
        """
        Returns the current trainable weights in a dictionary.
        Dict keys are layer names and dict values are the layers in numpy format.
        Could be named like this: {'conv-1': w, ... }
        """

        weights = {}
        for l in self._weight_layers:
            layer = self._get_layer(l)

            weights[l] = layer.get_weights()[0]


        return weights

    def get_layer_names(self):
        """
        Returns the relevant layer names in an array. This can be used to set pruning values
        :return: Array, eg. ['conv-1', 'conv-2', ... 'dense-1', ...]
        """
        return self._weight_layers

    def _get_layer(self, name):
        """
        Helper for getting a layer from the underlying model
        :param name: String
        :return: keras layer
        """
        return self._model.get_layer(
             ("prune_low_magnitude_" + name)
             if self._prune_percentages is not None and name in self._prune_percentages
             else name
        )



if __name__ == '__main__':
    ######
    # Example on how to use the model
    #####

    K = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')[:600]
    x_test = x_test.astype('float32')[:600]
    input_data_shape = x_train[0].shape


    y_train = keras.utils.to_categorical(y_train, K)[:600]
    y_test = keras.utils.to_categorical(y_test, K)[:600]

    conv = ConvModel(input_data_shape)

    conv.train(x_train, y_train, iterations=10)
    conv.evaluate(x_test, y_test)

    pruning = dict((name, 0.2) for name in conv.get_layer_names())
    conv.set_pruning(pruning)
    conv.train(x_train, y_train, iterations=10)
    conv.evaluate(x_test, y_test)

    pruning = dict((name, 0.4) for name in conv.get_layer_names())
    conv.set_pruning(pruning)
    conv.train(x_train, y_train, iterations=10)
    conv.evaluate(x_test, y_test)

    pruning = dict((name, 0.6) for name in conv.get_layer_names())
    conv.set_pruning(pruning)
    conv.train(x_train, y_train, iterations=10)
    conv.evaluate(x_test, y_test)