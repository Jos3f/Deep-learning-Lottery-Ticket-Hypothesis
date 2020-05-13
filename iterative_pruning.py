import numpy as np
import copy
import tensorflow as tf
from Lenet import Lenet
from ConvModel import ConvModel
from tensorflow.keras.datasets import cifar10
from tensorflow import keras

from datetime import datetime

class Model:
    """
    Model class with the functions needed for iterative pruning
    """

    network = None
    initial_weights = None  # Keep the very first weights (before training) to be able to reset the network
    prune_percentages = None

    def __init__(self, network, prune_percentages):
        """
        Args:
            network: Tensorflow network
            prune_percentages:  Prune percentage for each layer. Example: {'layer0': .2, 'layer1': .2, 'layer2': .1}
        """
        self.network = network
        self.prune_percentages = prune_percentages
        self.initial_weights = None  # TODO extract the initial weight from 'network' and store here
        return

    def reset(self):
        """
        Reset network to original weights.
        """
        # TODO
        return

    def get_trainable_weights(self):
        """
        returns the current trainable weights in a dictionary.
        Dict keys are layer names and dict values are the layers in numpy format.
        Could be named like this: {'layer0': w, 'layer1': w, 'layer2': w}
        """
        # TODO
        return {}

    def get_prune_percentages(self):
        """
        returns the prune percentages in a dictionary
        """
        return self.prune_percentages

    def train(self, masks=None):
        """
        Train the network with the mask if provided.
        May calculate loss and accuracy and store them somewhere for plotting.

        Args:
            masks: A dictionary mapped from layer name -> mask.
            Each mask is in numpy format and with the same dimensions as the weights.
            The keys are the same as for the dectinary keys returned by get_trainable_weights().
        """
        # TODO
        return


class IterativeTrainer:
    """
    Performs iterative pruning
    """

    iterations = 1  # Number of iterations
    model = None  # Store the Model object here
    pruning_percentages = None

    def __init__(self, model, iterations=1, pruning_percentages={}):
        """
        Args:
            model: The model we want to iteratively prune
            iterations: How many iterations we want to prune
        """
        self.model = model
        self.iterations = iterations
        self.pruning_percentages = pruning_percentages


    def train_iter(self, iterations, x_train, y_train, x_test, y_test, early_stopping=False, name="Model"):
        """
        Train iteratively
        """
        # dictionaries with our results:
        winning_result = {}

        now = datetime.now()
        now_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        result_file = "experiment_" + name + "_" + now_str + ".txt"

        # Train full model once first
        # self.model.train()
        layer_names = self.model.get_layer_names()

        print("Before pruning (iteration 0)")
        history = self.model.train(x_train, y_train, iterations, early_stopping)
        eval = self.model.evaluate(x_test, y_test)

        f = open("results/" + result_file, "a+")
        f.write("+Iteration 0\n+Early_stopping_at_epoch: {}\n+Train_loss: "
                "{}\n+Train_acc: {}\n+Test_loss: {}\n+Test_acc: {}".format(
            len(history.history['loss']), history.history["loss"][-1], history.history["accuracy"][-1],
            eval[0], eval[1]
        ))
        f.close()

        print("Random init")
        # train and evaluate a reinitialized model
        self.model.checkpoint_weights()
        self.model.set_pruning_random_init()
        history = self.model.train(x_train, y_train, iterations, early_stopping)
        eval = self.model.evaluate(x_test, y_test)
        self.model.reset_to_old_weights()

        f = open("results/" + result_file, "a+")
        f.write("\n\n#Iteration {}\n#Early_stopping_at_epoch: {}\n#Train_loss: "
                "{}\n#Train_acc: {}\n#Test_loss: {}\n#Test_acc: {}".format(0,
                                                                           len(history.history['loss']),
                                                                           history.history["loss"][-1],
                                                                           history.history["accuracy"][-1],
                                                                           eval[0], eval[1]
                                                                           ))
        f.close()



        # Initial masks are set to 1. This corresponds to not including all weights
        '''masks = {}
        layers = self.model.get_trainable_weights()
        for key, weights in layers.items():
            masks[key] = np.ones_like(weights)'''

        temp_pruning_percentages = self.pruning_percentages

        # Start the iterative pruning process
        for it in range(1, self.iterations + 1):
            print("\nIteration {}:".format(it))
            # Update pruning percentages
            temp_pruning_percentages = dict((name, self.update_percentages(it, self.pruning_percentages[name])) for name in self.pruning_percentages)

            print("Pruning percentages: {}".format(temp_pruning_percentages))

            # Update model pruning
            self.model.set_pruning(temp_pruning_percentages)

            # Train and evaluate model
            history = self.model.train(x_train, y_train, iterations, early_stopping)
            eval = self.model.evaluate(x_test, y_test)

            f = open("results/" + result_file, "a+")
            f.write("\n\n+Iteration {}\n+Pruning_percentages: {}\n+Early_stopping_at_epoch: {}\n+Train_loss: "
                    "{}\n+Train_acc: {}\n+Test_loss: {}\n+Test_acc: {}".format( it, temp_pruning_percentages,
                len(history.history['loss']), history.history["loss"][-1], history.history["accuracy"][-1],
                eval[0], eval[1]
            ))
            f.close()

            print("Random init")
            # train and evaluate a reinitialized model
            self.model.checkpoint_weights()
            self.model.set_pruning_random_init(temp_pruning_percentages)
            history = self.model.train(x_train, y_train, iterations, early_stopping)
            eval = self.model.evaluate(x_test, y_test)
            self.model.reset_to_old_weights()

            f = open("results/" + result_file, "a+")
            f.write("\n\n#Iteration {}\n#Pruning_percentages: {}\n#Early_stopping_at_epoch: {}\n#Train_loss: "
                    "{}\n#Train_acc: {}\n#Test_loss: {}\n#Test_acc: {}".format( it, temp_pruning_percentages,
                len(history.history['loss']), history.history["loss"][-1], history.history["accuracy"][-1],
                eval[0], eval[1]
            ))
            f.close()

            '''# Update masks by pruning
            masks = self.prune(masks, layers)
            # Train with our new masking
            self.model.reset()
            # self.model.train()
            self.model.train(x_train, y_train, x_test,  y_test, masks)
            # Get the new updated layer weights
            layers = self.model.get_trainable_weights()'''


    def prune(self, masks, layers):
        """
        Prune the network by pruning the remaining weights in the layers
        """
        # Copy previous masks
        updated_masks = copy.deepcopy(masks)
        for layer_name in layers:
            # Calculate how many weights we need to prune
            active_nodes = (masks[layer_name] == 1)  # indices of active nodes
            prune_percentages = self.model.get_prune_percentages()
            prune_count = int(round(prune_percentages[layer_name] * np.sum(active_nodes)))

            # Sort by the magnitude of the weights for the weights in our mask and extract the element value where
            # we have gone past 'prune_count' nodes.
            cutoff_value = np.sort(np.absolute(layers[layer_name][active_nodes]))[prune_count]

            # Update mask by disabling the nodes with magnitude smaller than/equal to our cutoff value
            magnitude_under_cutoff = (np.absolute(layers[layer_name]) <= cutoff_value)
            updated_masks[layer_name][magnitude_under_cutoff] = 0

        return updated_masks

    def update_percentages(self, iteration, percentage_each_it):
        new_percentage = 1 - (1 - percentage_each_it)**iteration
        return new_percentage


def lenetModelExperiment():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    lenet = Lenet()
    pruning = dict((name, 0.2) for name in lenet.get_layer_names())
    iter_trainer = IterativeTrainer(lenet, iterations=20, pruning_percentages=pruning)
    iterations = 5 * len(x_train) / 60 # 5000
    iterations = 50000
    iter_trainer.train_iter(iterations, x_train, y_train, x_test, y_test, early_stopping=True, name="lenet")
    return


def convNetExperiment():
    K = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')#[:600]
    x_test = x_test.astype('float32')#[:600]
    input_data_shape = x_train[0].shape

    y_train = keras.utils.to_categorical(y_train, K)#[:600]
    y_test = keras.utils.to_categorical(y_test, K)#[:600]

    conv = ConvModel(input_data_shape)
    pruning = dict((name, 0.2) if name[0] == 'd' else (name, 0.15) for name in conv.get_layer_names())
    iter_trainer = IterativeTrainer(conv, iterations=20, pruning_percentages=pruning)
    # TODO turning off early stopping may improve results. Stops too early.
    iter_trainer.train_iter(30000, x_train, y_train, x_test,  y_test, early_stopping=True, name="conv")
    return


def main():
    '''Begin with lenet'''
    #lenetModelExperiment()

    '''Experiment with Convmodel'''
    convNetExperiment()

    # 1: Create model
    # 2: Create IterativeTrainer with model
    # 3: Call IterativeTrainer.train_iter()
    return


if __name__ == '__main__':
    main()
