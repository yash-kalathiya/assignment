"""
Neural Networks HW - Task 2: Neural Network Implementation

This module implements a 2-layer neural network with gradient descent training.
"""

import numpy as np

class NeuralNetwork:
    """2-layer neural network with tanh activation."""
    
    def __init__(self, num_hidden_units=5, learning_rate=0.01, random_seed=42):
        np.random.seed(random_seed)
        self.m = num_hidden_units
        self.lr = learning_rate
        
        # Xavier initialization for better convergence
        fan_in_w1, fan_out_w1 = 3, self.m
        xavier_std_w1 = np.sqrt(2.0 / (fan_in_w1 + fan_out_w1))
        self.W1 = np.random.normal(0, xavier_std_w1, (3, self.m))
        
        fan_in_w2, fan_out_w2 = self.m + 1, 1
        xavier_std_w2 = np.sqrt(2.0 / (fan_in_w2 + fan_out_w2))
        self.W2 = np.random.normal(0, xavier_std_w2, (self.m + 1, 1))
        
        # Store intermediate values
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.X = None
        
    def forward_propagation(self, X):
        """Forward pass through network."""
        n_samples = X.shape[0]
        
        # Add bias to input
        X_with_bias = np.column_stack([np.ones(n_samples), X])
        self.X = X_with_bias
        
        # Hidden layer
        self.z1 = X_with_bias @ self.W1
        self.a1 = np.tanh(self.z1)
        
        # Output layer
        a1_with_bias = np.column_stack([np.ones(n_samples), self.a1])
        self.z2 = a1_with_bias @ self.W2
        self.a2 = self.z2  # Linear output
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        """Compute squared error loss."""
        return 0.5 * np.mean((y_true - y_pred) ** 2)
    
    def back_propagation(self, y_true):
        """Compute gradients using backpropagation."""
        n_samples = y_true.shape[0]
        
        # Output layer gradients
        dz2 = (self.a2 - y_true) / n_samples
        
        # W2 gradients
        a1_with_bias = np.column_stack([np.ones(n_samples), self.a1])
        dW2 = a1_with_bias.T @ dz2
        
        # Hidden layer gradients
        da1 = dz2 @ self.W2[1:].T
        dz1 = da1 * (1 - np.tanh(self.z1) ** 2)
        
        # W1 gradients
        dW1 = self.X.T @ dz1
        
        return dW1, dW2
    
    def update_weights(self, dW1, dW2):
        """Update weights using gradient descent."""
        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2
    
    def train_step(self, X, y):
        """One training step."""
        predictions = self.forward_propagation(X)
        loss = self.compute_loss(y, predictions)
        dW1, dW2 = self.back_propagation(y)
        self.update_weights(dW1, dW2)
        return loss
    
    def predict(self, X):
        """Make predictions."""
        return self.forward_propagation(X)

def train_network(X, y, num_hidden_units=5, learning_rate=0.01, epochs=1000, 
                 batch_size=None, verbose=True):
    """Train neural network with gradient descent."""
    
    network = NeuralNetwork(num_hidden_units, learning_rate)
    loss_history = []
    n_samples = X.shape[0]
    
    if batch_size is None:
        batch_size = n_samples
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Shuffle for stochastic training
        if batch_size < n_samples:
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
        else:
            X_shuffled = X
            y_shuffled = y
        
        # Process batches
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X_shuffled[i:end_idx]
            y_batch = y_shuffled[i:end_idx]
            
            batch_loss = network.train_step(X_batch, y_batch)
            epoch_losses.append(batch_loss)
        
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        
        if verbose and (epoch + 1) % (epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return network, loss_history

if __name__ == "__main__":
    print("Task 2: Neural Network Implementation")
    
    # Test 1: XOR Problem
    print("\\nTest 1: XOR Problem")
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]], dtype=float)
    
    network_xor, _ = train_network(X_xor, y_xor, num_hidden_units=4, 
                                  learning_rate=0.1, epochs=2000, verbose=False)
    
    predictions_xor = network_xor.predict(X_xor)
    print("XOR Results:")
    for i in range(len(X_xor)):
        print(f"Input: {X_xor[i]} -> Target: {y_xor[i][0]:.0f}, Predicted: {predictions_xor[i][0]:.4f}")
    
    # Test 2: Regression with Stochastic Gradient Descent
    print("\\nTest 2: Regression Problem (SGD)")
    np.random.seed(42)
    X_reg = np.random.randn(400, 2)
    y_reg = (np.sin(X_reg[:, 0]) + np.cos(X_reg[:, 1]) + 0.1 * np.random.randn(400)).reshape(-1, 1)
    
    network_reg, loss_history = train_network(X_reg, y_reg, num_hidden_units=10, 
                                             epochs=1000, batch_size=1)  
    
    final_loss = network_reg.compute_loss(y_reg, network_reg.predict(X_reg))
    print(f"Final training loss: {final_loss:.6f}")