import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import tensorflow_model_optimization as tfmot


class Lenet:
    """
    Model class with the functions needed for iterative pruning
    """

    network = None
    initial_weights = None  # Keep the very first weights (before training) to be able to reset the network
    prune_percentages = None

    def __init__(self, use_dropout=False):

        # Hyper-params that can be changed
        self.K = 10
        self.learning_rate = 1.2 * (10 ** -3)
        self.batch_size = 60

        self._use_dropout = use_dropout

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

        self._initial_random_weights = None
        self._last_used_masks = None

        return

    def _build(self, inp=(28, 28)):
        """
        Builds the model from scratch. If we have set the pruning percentages we will build the model with pruning.
        """

        self._weight_layers = []
        self._model = Sequential()

        self._add_layer(Flatten(input_shape=inp))
        self._add_layer(Dense(300, name='dense-1', activation="relu", kernel_initializer="glorot_normal"))
        if self._use_dropout: self._add_layer(Dropout(0.2, name='dropout-1'))
        self._add_layer(Dense(100, name='dense-2', activation="relu", kernel_initializer="glorot_normal"))
        if self._use_dropout: self._add_layer(Dropout(0.1, name='dropout-2'))
        self._add_layer(Dense(10, name='dense-3', activation="softmax", kernel_initializer="glorot_normal"))
        self._weight_layers.extend(['dense-1', 'dense-2', 'dense-3'])

    def _compile(self):
        opt = tf.keras.optimizers.Adam(lr=self.learning_rate)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self._model.compile(optimizer=opt,
                            loss=loss_fn,
                            metrics=['accuracy'])

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

    def evaluate(self, x, y):
        """
        Evaluate a model on test data
        :param x: input
        :param y: labels
        """
        eval = self._model.evaluate(x, y)
        print("Accuracy: " + str(eval[1]) + ", Loss: " + str(eval[0]))
        return eval

    def train(self, x_train, y_train, iterations=50000, early_stopping=False):
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
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=1)
            callbacks.append(early_stopping)
            val_split = 0.2

        epochs = int(iterations * self.batch_size / (x_train.shape[0] * (1 - val_split) ))
        return self._model.fit(x_train, y_train, self.batch_size, epochs, callbacks=callbacks, validation_split=val_split)

    def _get_trainable_weights(self):
        """
        returns the current trainable weights in a dictionary.
        Dict keys are layer names and dict values are the layers in numpy format.
        Could be named like this: {'layer0': w, 'layer1': w, 'layer2': w}
        """
        weights = {}
        for l in self._weight_layers:
            layer = self._get_layer(l)

            weights[l] = layer.get_weights()[0]
        return weights

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
        the unpruned weights to random weights.
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

    def get_prune_percentages(self):
        """
        returns the prune percentages in a dictionary
        """
        return self.prune_percentages

    def get_layer_names(self):
        """
        Returns the relevant layer names in an array. This can be used to set pruning values
        :return: Array, eg. ['dense-1', 'dense-2', ... 'dense-k', ...]
        """
        return self._weight_layers

    def _get_layer(self, name):
        """
        Helper for getting a layer from the underlying model
        :param name: String
        :return: keras layer
        """
        print(name + " " + str(self._prune_percentages))
        return self._model.get_layer(
            ("prune_low_magnitude_" + name)
            if self._prune_percentages is not None and name in self._prune_percentages
            else name
        )