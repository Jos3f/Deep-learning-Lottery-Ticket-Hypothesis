import numpy as np
import copy


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

    def __init__(self, model, iterations=1):
        """
        Args:
            model: The model we want to iteratively prune
            iterations: How many iterations we want to prune
        """
        self.model = model
        self.iterations = iterations


    def train_iter(self, iterations):
        """
        Train iteratively
        """
        # Train full model once first
        self.model.train()

        # Initial masks are set to 1. This corresponds to not including all weights
        masks = {}
        layers = self.model.get_trainable_weights()
        for key, weights in layers:
            masks[key] = np.ones_like(weights)

        # Start the iterative pruning process
        for it in range(iterations):
            # Update masks by pruning
            masks = self.prune(masks, layers)
            # Train with our new masking
            self.model.reset()
            self.model.train(masks)
            # Get the new updated layer weights
            layers = self.model.get_trainable_weights()


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
            prune_count = int(round(prune_percentages[layer_name] * active_nodes.size))

            # Sort by the magnitude of the weights for the weights in our mask and extract the element value where
            # we have gone past 'prune_count' nodes.
            cutoff_value = np.sort(np.absolute(layers[layer_name][active_nodes]))[prune_count]

            # Update mask by disabling the nodes with magnitude smaller than/equal to our cutoff value
            magnitude_under_cutoff = (np.absolute(layers[layer_name]) <= cutoff_value)
            updated_masks[layer_name][magnitude_under_cutoff] = 0

            return updated_masks


def main():
    # 1: Create model
    # 2: Create IterativeTrainer with model
    # 3: Call IterativeTrainer.train_iter()
    return


if __name__ == '__main__':
    main()
