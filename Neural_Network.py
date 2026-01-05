import numpy as np
"""
input_num = number of inputs to the neuron

"""
class Neuron: 
    #Initialize neuron with random weights and bias
    
    def __init__(self, input_num):
        self.weights = np.random.uniform(-1, 1, input_num)
        self.bias = np.random.uniform()
    
    # Calculates the wx + b
    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(np.dot(self.weights, inputs) + self.bias)
        return self.output
    
    # Transfer Function (1/1+e^-x) for non linearity
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Sensitivity of the neuron output to changes in input
    def sigmoid_derivative(self):
        return self.output * (1 - self.output)
    
    # Update weights in the direction of reucing the error
    def update_weights(self, learning_rate, delta):
        self.weights += learning_rate * delta * self.inputs
        self.bias += learning_rate * delta

class Layer:
    # Initialize layer with a number of neurons
    def __init__(self, num_neurons, input_num):
        self.neurons = [Neuron(input_num) for _ in range(num_neurons)]

    # Forward pass through all neurons in the layer using Neuron.forward
    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])
    
    # Backward pass to update weights based on errors
    # Implements the backpropagation algorithm (chain rule) 
    def backward(self, errors, learning_rate):
        deltas = []
        for i, neuron in enumerate(self.neurons):
            delta = errors[i] * neuron.sigmoid_derivative()
            neuron.update_weights(learning_rate, delta)
            deltas.append(delta)
        return np.dot(np.array([neuron.weights for neuron in self.neurons]).T, deltas)
    
class NeuralNetwork:

    def __init__(self, layers, learning_rate, epochs):
        self.layers = []
        self.learning_rate = learning_rate
        self.epochs = epochs

        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i + 1], layers[i]))

    def train(self, inputs, output):
        for epoch in range(self.epochs):
            
            error_sum = 0
            for x, y in zip(inputs, output):

                # 1.Run forward pass
                current_signal = x
                for layer in self.layers:
                    current_signal = layer.forward(current_signal)
                prediction = current_signal

                # 2. Compute error
                error = y - prediction

                # 3. Backpropagate error and update weights
                for layer in reversed(self.layers):
                    error = layer.backward(error, self.learning_rate)
                error_sum += np.mean(error**2)
            print(f"Epoch {epoch}: Mean Squared Error: {error_sum/len(inputs)}")
                

    def final_predict(self, inputs):

        # processes inputs through all layers to get final output
        activations = inputs
        for layer in self.layers:
            activations = layer.forward(activations)
        
        if activations[0] >= 0.5:
            print("1")

        else: 
            print("0")

        return activations


    



        

    

    





