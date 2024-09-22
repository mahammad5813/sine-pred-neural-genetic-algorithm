import numpy as np
import random
from nn_funcs import array_tweak
from activations import relu, sigmoid, tanh, leaky_relu, elu_array

class NN():
    def __init__(self, shape, weights=None, biases=None):
        """
        shape = (n_layer1, n_layer2 ... n_layern)
        """
        if weights and biases:
            self.weights = weights
            self.biases = biases
        else:
            self.weights = [np.random.uniform(-1,1,(shape[i-1], shape[i])) for i in range(1,len(shape))]
            self.biases = [np.random.uniform(-1,1,(shape[i])) for i in range(1, len(shape))]
            

        self.score = 0
        self.shape = shape

    def mutate(self):

        for i,w_layer in enumerate(self.weights):
            self.weights[i] = array_tweak(w_layer)

        for i,b_layer in enumerate(self.biases):
            self.biases[i] = array_tweak(b_layer)

    def evaluate(self, inputs):

        for i in range(len(self.weights)):
            inputs = np.dot(inputs, self.weights[i])+self.biases[i]
            if i < len(self.weights)-1:
                inputs = leaky_relu(inputs)
            # else:
            #     inputs = tanh(inputs)

        return inputs

    def evaluate_bunch(self, bunch_inputs):
        
        output = np.apply_along_axis(self.evaluate, 1, bunch_inputs)
        return output
