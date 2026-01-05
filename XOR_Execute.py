import numpy as np
from Neural_Network import NeuralNetwork
from data_gen import xor_generator
def main():
 
    inputs = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    # Initialize the structure of the neural network
    layers = [2, 8, 8, 1]  # Input layer, two hidden layers, output layer

    nn = NeuralNetwork(layers=layers, learning_rate=1e-1, epochs=60)

    train_data = list(xor_generator(inputs, outputs, n_samples=2000))
    
    # We can zip these back into the format your current train() expects
    train_inputs, train_targets = zip(*train_data)


    nn.train(train_inputs, train_targets)
    
    # Test the results

    for x, output in zip(inputs, outputs):
        prediction = nn.final_predict(x)

        print(f"Input: {x} | Predicted: {prediction} | Should be: {output[0]}")

if __name__ == "__main__":
    main()

