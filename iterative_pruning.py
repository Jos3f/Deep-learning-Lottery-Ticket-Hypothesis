import numpy as np
import copy
import tensorflow as tf
from Lenet import Lenet
from ConvModel import ConvModel
from tensorflow.keras.datasets import cifar10
from tensorflow import keras

from datetime import datetime

class IterativeTrainer:
    """
    Performs iterative pruning on a model and evaluates it.
    """

    iterations = 1  # Number of iterations
    model = None  # Store the Model object here
    pruning_percentages = None # How much each layer should be pruned each layer

    def __init__(self, model, iterations=1, pruning_percentages={}):
        """
        Args:
            model: The model we want to iteratively prune
            iterations: How many iterations we want to prune
            pruning_percentages: How much we should prune each layer.
        """
        self.model = model
        self.iterations = iterations
        self.pruning_percentages = pruning_percentages


    def train_iter(self, iterations, x_train, y_train, x_test, y_test, early_stopping=False, name="Model"):
        """
        Train iteratively and evaluate each iteration by comparing it to a randomly initialized model with the same
        architecture and pruning. Evaluation is stored in a separate file.
        Args:
            iterations: Number of max training iterations
            x_train: training input data
            y_train: training labels
            x_test: test input data
            y_test: test labels
            early_stopping: Use early stopping each training session
            name: Name of the output file
        """
        # dictionaries with our results:
        winning_result = {}

        # Timestamp used for filename
        now = datetime.now()
        now_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        # File name
        result_file = "experiment_" + name + "_" + now_str + ".txt"

        # Train full model once first, without pruning
        print("Before pruning (iteration 0)")
        history = self.model.train(x_train, y_train, iterations, early_stopping)
        eval = self.model.evaluate(x_test, y_test)

        # Update results file
        f = open("results/" + result_file, "a+")
        f.write("+Iteration 0\n+Early_stopping_at_epoch: {}\n+Train_loss: "
                "{}\n+Train_acc: {}\n+Test_loss: {}\n+Test_acc: {}".format(
            len(history.history['loss']), history.history["loss"][-1], history.history["accuracy"][-1],
            eval[0], eval[1]
        ))
        f.close()

        # Train and evaluate a reinitialized model, also without pruning. Optional.
        print("Random init")
        self.model.checkpoint_weights()
        self.model.set_pruning_random_init()
        history = self.model.train(x_train, y_train, iterations, early_stopping)
        eval = self.model.evaluate(x_test, y_test)
        self.model.reset_to_old_weights()

        # Update results file
        f = open("results/" + result_file, "a+")
        f.write("\n\n#Iteration {}\n#Early_stopping_at_epoch: {}\n#Train_loss: "
                "{}\n#Train_acc: {}\n#Test_loss: {}\n#Test_acc: {}".format(0,
                                                                           len(history.history['loss']),
                                                                           history.history["loss"][-1],
                                                                           history.history["accuracy"][-1],
                                                                           eval[0], eval[1]
                                                                           ))
        f.close()

        # Start the iterative pruning process
        for it in range(1, self.iterations + 1):
            print("\nIteration {}:".format(it))
            # Update pruning percentages
            temp_pruning_percentages = dict((name, self.update_percentages(it, self.pruning_percentages[name])) for name in self.pruning_percentages)

            print("Pruning percentages: {}".format(temp_pruning_percentages))

            # Update model to have the correct architecture
            self.model.set_pruning(temp_pruning_percentages)

            # Train and evaluate model
            history = self.model.train(x_train, y_train, iterations, early_stopping)
            eval = self.model.evaluate(x_test, y_test)

            # Update results file
            f = open("results/" + result_file, "a+")
            f.write("\n\n+Iteration {}\n+Pruning_percentages: {}\n+Early_stopping_at_epoch: {}\n+Train_loss: "
                    "{}\n+Train_acc: {}\n+Test_loss: {}\n+Test_acc: {}".format( it, temp_pruning_percentages,
                len(history.history['loss']), history.history["loss"][-1], history.history["accuracy"][-1],
                eval[0], eval[1]
            ))
            f.close()

            print("Random init")
            # train and evaluate a reinitialized model (using our random initialisation)
            # with the same architecture as before
            self.model.checkpoint_weights() # Create a check point for the weights that will be used for calculating the pruning next iteration.
            self.model.set_pruning_random_init(temp_pruning_percentages)
            history = self.model.train(x_train, y_train, iterations, early_stopping)
            eval = self.model.evaluate(x_test, y_test)
            self.model.reset_to_old_weights() # Resore model to the weights in our check point

            # Update results file
            f = open("results/" + result_file, "a+")
            f.write("\n\n#Iteration {}\n#Pruning_percentages: {}\n#Early_stopping_at_epoch: {}\n#Train_loss: "
                    "{}\n#Train_acc: {}\n#Test_loss: {}\n#Test_acc: {}".format( it, temp_pruning_percentages,
                len(history.history['loss']), history.history["loss"][-1], history.history["accuracy"][-1],
                eval[0], eval[1]
            ))
            f.close()


    def update_percentages(self, iteration, percentage_each_it):
        """Calculates how much a layer should be pruned this iteration
        Args:
            iteration: The current iteration
            percentage_each_it: How much we wish to prune each iteration
        """
        new_percentage = 1 - (1 - percentage_each_it)**iteration
        return new_percentage


def lenetModelExperiment():
    """
    Run experiment on the lenet architecture with MNIST data set
    """

    # Prepare data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Create model
    lenet = Lenet()
    pruning = dict((name, 0.2) for name in lenet.get_layer_names()) # Create pruning schedule
    iter_trainer = IterativeTrainer(lenet, iterations=20, pruning_percentages=pruning)
    iterations = 5 * len(x_train) / 60 # 5000
    iterations = 50000
    # Start experiment
    iter_trainer.train_iter(iterations, x_train, y_train, x_test, y_test, early_stopping=True, name="lenet")
    return


def convNetExperiment():
    """
    Run experiment on the Conv-6 architecture with CIFAR10 data set
    """

    # Prepare data
    K = 10 # Number of outputs
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')#[:600]
    x_test = x_test.astype('float32')#[:600]
    input_data_shape = x_train[0].shape

    y_train = keras.utils.to_categorical(y_train, K)#[:600]
    y_test = keras.utils.to_categorical(y_test, K)#[:600]

    # Create model
    conv = ConvModel(input_data_shape)
    # Create pruning schedule
    pruning = dict((name, 0.2) if name[0] == 'd' else (name, 0.15) for name in conv.get_layer_names())
    iter_trainer = IterativeTrainer(conv, iterations=20, pruning_percentages=pruning)
    # Start experiment
    iter_trainer.train_iter(30000, x_train, y_train, x_test,  y_test, early_stopping=True, name="conv")
    return


def lenetFashionExperiment():
    """
    Run experiment on the lenet architecture with MNIST fashion data set
    """

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    lenet = Lenet()
    pruning = dict((name, 0.2) for name in lenet.get_layer_names())
    iter_trainer = IterativeTrainer(lenet, iterations=20, pruning_percentages=pruning)
    iter_trainer.train_iter(50000, x_train, y_train, x_test, y_test, early_stopping=True, name="fashion")
    return

def main():
    '''Experiment with lenetmodel'''
    #lenetModelExperiment()

    '''Experiment with Convmodel'''
    convNetExperiment()

    '''Experiment with another dataset with lenet model'''
    #lenetFashionExperiment()
    return


if __name__ == '__main__':
    main()
