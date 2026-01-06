# XOR Neural Network From Scratch

This is a NumPY-based implementation of a Multi-Layer Perceptroj (MLP) built without high-level ML libraries like PyTorch or TensorFlow

1. Key Features 

Object Oriented Design: To create the Neural Network classes were created for the individual aspects - Nodes, Layers and the Neural Network itself. This allows the ability to change the overalll structure of the NN. 

Custom Backpropagation: To teach the desired behaviour of an XOR gate, the chain rule was implemented manually to change the weightings of the nodes during each epoch. 

Stochastic Data Generation: The project uses a custom generator for the listing of the inputs and outputs to produce a training set of thousands of samples to achieve the necessary variance for the Gradient Descent algorithm. 

2. Repository Structure

Neural_Network.py: The core architecture which includes the Activation functions and forward/backward passes

data_gen.py: Data utility for generating training samples

XOR_Execute.py: The entry point for training and evaluation, which includes the parameters (Layers, learning rate, epochs and samples) to tune the implementation of the architecture.

3. Mathematical Implementation

Activation Function: Sigmoid and Sigmoid Derivative
To solve the XOR problem, the network must learn a non-linear decision boundary. We use the Sigmoid function to "squash" the output of each neuron into a probability range between 0 and 1.

The derivative is calculated during the backward pass to determine the sensitivity of the output to changes in the weighted sum.

Loss Function: Mean Squared Error (MSE)

We quantify the network's performance by measuring the average squared difference between the predicted output and the true XOR target.

Optimization: Gradient Descent & Learning Rate

The network learns by navigating the "loss landscape" to find the global minimum. We use Stochastic Gradient Descent (SGD) to update each weight ($w$) in the opposite direction of the gradient:

The Learning Rate $\eta$ is the step size. For this implementation, a learning rate of 0.1 was chosen to balance convergence speed and stability.

4. Results

During training, the Mean Squared Error typically spikes initially as the network explores the weight space, before converging toward zero.
