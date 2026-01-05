import numpy as np
def xor_generator(inputs, outputs, n_samples):
    """

    Randomly samples n pairs from the provided XOR distribution.
    """
    # Get the number of available patterns (which is 4)
    num_patterns = len(inputs)
    
    # Generate n_samples worth of random indices
    # This is like doing rand() % 4, n times
    indices = np.random.choice(num_patterns, size=n_samples)
    
    # Yield each pair one by one
    for i in indices:
        yield inputs[i], outputs[i]