"""
Deep Learning Model - Task 2: Two-Layer Network Architecture

Implementation of a multi-layer perceptron using backpropagation algorithm.
Author: Neural Network Lab
Date: October 2025
"""

import numpy as np


class TwoLayerPerceptron:
    """
    A feed-forward neural network with two layers.
    
    Architecture:
        - Input layer with bias term
        - Hidden layer with hyperbolic tangent activation
        - Output layer with linear activation
    """
    
    def __init__(
        self, 
        hidden_nodes=5, 
        alpha=0.01, 
        seed_value=42
    ):
        """
        Initialize the two-layer perceptron.
        
        Args:
            hidden_nodes: Number of neurons in the hidden layer
            alpha: Learning rate for gradient descent optimization
            seed_value: Random seed for reproducibility
        """
        np.random.seed(seed_value)
        
        self.hidden_nodes = hidden_nodes
        self.alpha = alpha
        
        # Initialize weights using Xavier/Glorot initialization
        input_dimension = 3  # 2 features + bias
        
        # First layer weight matrix (input -> hidden)
        fan_input_first = input_dimension
        fan_output_first = self.hidden_nodes
        std_dev_first = np.sqrt(2.0 / (fan_input_first + fan_output_first))
        self.weight_matrix_layer1 = np.random.normal(
            loc=0.0, 
            scale=std_dev_first, 
            size=(input_dimension, self.hidden_nodes)
        )
        
        # Second layer weight matrix (hidden -> output)
        fan_input_second = self.hidden_nodes + 1
        fan_output_second = 1
        std_dev_second = np.sqrt(2.0 / (fan_input_second + fan_output_second))
        self.weight_matrix_layer2 = np.random.normal(
            loc=0.0,
            scale=std_dev_second,
            size=(self.hidden_nodes + 1, 1)
        )
        
        # Cache for intermediate computations
        self.hidden_pre_activation = None
        self.hidden_post_activation = None
        self.output_pre_activation = None
        self.output_post_activation = None
        self.input_with_bias = None
        
    def forward_pass(self, input_data):
        """
        Perform forward propagation through the network.
        
        Args:
            input_data: Input feature matrix (n_samples x n_features)
            
        Returns:
            Network predictions
        """
        num_samples = input_data.shape[0]
        
        # Augment input with bias column
        bias_column = np.ones((num_samples, 1))
        self.input_with_bias = np.column_stack([bias_column, input_data])
        
        # Compute hidden layer activations
        self.hidden_pre_activation = self.input_with_bias @ self.weight_matrix_layer1
        self.hidden_post_activation = np.tanh(self.hidden_pre_activation)
        
        # Compute output layer activations
        hidden_with_bias = np.column_stack([bias_column, self.hidden_post_activation])
        self.output_pre_activation = hidden_with_bias @ self.weight_matrix_layer2
        self.output_post_activation = self.output_pre_activation  # Linear activation
        
        return self.output_post_activation
    
    
    def calculate_loss(self, target_values, predicted_values):
        """
        Calculate mean squared error loss.
        
        Args:
            target_values: True target values
            predicted_values: Model predictions
            
        Returns:
            MSE loss value
        """
        squared_errors = (target_values - predicted_values) ** 2
        return 0.5 * np.mean(squared_errors)
    
    def backward_pass(self, target_values):
        """
        Compute gradients via backpropagation algorithm.
        
        Args:
            target_values: True target values
            
        Returns:
            Tuple of (gradient_layer1, gradient_layer2)
        """
        num_samples = target_values.shape[0]
        
        # Calculate output layer error gradient
        output_error_gradient = (self.output_post_activation - target_values) / num_samples
        
        # Compute gradients for second layer weights
        bias_column = np.ones((num_samples, 1))
        hidden_with_bias = np.column_stack([bias_column, self.hidden_post_activation])
        gradient_weight_layer2 = hidden_with_bias.T @ output_error_gradient
        
        # Backpropagate error to hidden layer
        hidden_error = output_error_gradient @ self.weight_matrix_layer2[1:].T
        
        # Compute derivative of tanh activation
        tanh_derivative = 1 - np.square(np.tanh(self.hidden_pre_activation))
        hidden_delta = hidden_error * tanh_derivative
        
        # Compute gradients for first layer weights
        gradient_weight_layer1 = self.input_with_bias.T @ hidden_delta
        
        return gradient_weight_layer1, gradient_weight_layer2
    
    
    def apply_gradient_update(self, gradient_layer1, gradient_layer2):
        """
        Update network weights using computed gradients.
        
        Args:
            gradient_layer1: Gradient for first layer weights
            gradient_layer2: Gradient for second layer weights
        """
        self.weight_matrix_layer1 -= self.alpha * gradient_layer1
        self.weight_matrix_layer2 -= self.alpha * gradient_layer2
    
    def perform_training_iteration(self, input_data, target_values):
        """
        Execute one complete training iteration.
        
        Args:
            input_data: Input feature matrix
            target_values: True target values
            
        Returns:
            Loss value for this iteration
        """
        # Forward pass
        predictions = self.forward_pass(input_data)
        
        # Calculate loss
        iteration_loss = self.calculate_loss(target_values, predictions)
        
        # Backward pass
        grad_layer1, grad_layer2 = self.backward_pass(target_values)
        
        # Update parameters
        self.apply_gradient_update(grad_layer1, grad_layer2)
        
        return iteration_loss
    
    
    def generate_predictions(self, input_data):
        """
        Generate predictions for new data.
        
        Args:
            input_data: Input feature matrix
            
        Returns:
            Model predictions
        """
        return self.forward_pass(input_data)


def execute_training_procedure(
    feature_matrix, 
    target_vector, 
    hidden_layer_size=5, 
    learning_rate_param=0.01, 
    training_epochs=1000, 
    mini_batch_size=None, 
    show_progress=True
):
    """
    Train a two-layer neural network using gradient descent.
    
    Args:
        feature_matrix: Input features (n_samples x n_features)
        target_vector: Target values (n_samples x 1)
        hidden_layer_size: Number of hidden layer neurons
        learning_rate_param: Step size for gradient descent
        training_epochs: Number of complete passes through data
        mini_batch_size: Size of mini-batches (None for full batch)
        show_progress: Whether to print training progress
        
    Returns:
        Tuple of (trained_model, loss_trajectory)
    """
    
    # Initialize network
    trained_model = TwoLayerPerceptron(
        hidden_nodes=hidden_layer_size, 
        alpha=learning_rate_param
    )
    
    loss_trajectory = []
    total_samples = feature_matrix.shape[0]
    
    # Set batch size
    if mini_batch_size is None:
        mini_batch_size = total_samples
    
    # Training loop
    for epoch_idx in range(training_epochs):
        epoch_loss_values = []
        
        # Data shuffling for stochastic optimization
        if mini_batch_size < total_samples:
            shuffle_indices = np.random.permutation(total_samples)
            features_shuffled = feature_matrix[shuffle_indices]
            targets_shuffled = target_vector[shuffle_indices]
        else:
            features_shuffled = feature_matrix
            targets_shuffled = target_vector
        
        # Mini-batch processing
        for batch_start in range(0, total_samples, mini_batch_size):
            batch_end = min(batch_start + mini_batch_size, total_samples)
            
            feature_batch = features_shuffled[batch_start:batch_end]
            target_batch = targets_shuffled[batch_start:batch_end]
            
            batch_loss_value = trained_model.perform_training_iteration(
                feature_batch, 
                target_batch
            )
            epoch_loss_values.append(batch_loss_value)
        
        # Record average epoch loss
        mean_epoch_loss = np.mean(epoch_loss_values)
        loss_trajectory.append(mean_epoch_loss)
        
        # Display progress
        if show_progress and (epoch_idx + 1) % (training_epochs // 10) == 0:
            print(f"Epoch {epoch_idx + 1}/{training_epochs}, Loss: {mean_epoch_loss:.6f}")
    
    return trained_model, loss_trajectory


if __name__ == "__main__":
    print("=" * 60)
    print("Task 2: Two-Layer Neural Network Implementation")
    print("=" * 60)
    
    # ========== Experiment 1: XOR Classification Problem ==========
    print("\n" + "=" * 60)
    print("Experiment 1: XOR Logic Gate Classification")
    print("=" * 60)
    
    # Prepare XOR dataset
    xor_features = np.array([
        [0, 0], 
        [0, 1], 
        [1, 0], 
        [1, 1]
    ])
    xor_targets = np.array([
        [0], 
        [1], 
        [1], 
        [0]
    ], dtype=float)
    
    # Train XOR model
    xor_model, xor_loss_history = execute_training_procedure(
        feature_matrix=xor_features,
        target_vector=xor_targets,
        hidden_layer_size=4,
        learning_rate_param=0.1,
        training_epochs=2000,
        show_progress=False
    )
    
    # Evaluate XOR model
    xor_predictions = xor_model.generate_predictions(xor_features)
    
    print("\nXOR Classification Results:")
    print("-" * 60)
    print(f"{'Input':<15} {'Expected':<15} {'Predicted':<15}")
    print("-" * 60)
    for sample_idx in range(len(xor_features)):
        input_str = str(xor_features[sample_idx])
        expected_val = f"{xor_targets[sample_idx][0]:.0f}"
        predicted_val = f"{xor_predictions[sample_idx][0]:.4f}"
        print(f"{input_str:<15} {expected_val:<15} {predicted_val:<15}")
    
    # ========== Experiment 2: Nonlinear Regression with SGD ==========
    print("\n" + "=" * 60)
    print("Experiment 2: Nonlinear Regression (Stochastic GD)")
    print("=" * 60)
    
    # Generate synthetic regression dataset
    np.random.seed(42)
    num_regression_samples = 400
    regression_features = np.random.randn(num_regression_samples, 2)
    
    # Create complex nonlinear target function
    regression_targets = (
        np.sin(regression_features[:, 0]) + 
        np.cos(regression_features[:, 1]) + 
        0.1 * np.random.randn(num_regression_samples)
    ).reshape(-1, 1)
    
    # Train regression model with SGD (batch_size=1)
    regression_model, regression_loss_history = execute_training_procedure(
        feature_matrix=regression_features,
        target_vector=regression_targets,
        hidden_layer_size=10,
        training_epochs=1000,
        mini_batch_size=1  # Stochastic Gradient Descent
    )
    
    # Calculate final performance metrics
    final_predictions = regression_model.generate_predictions(regression_features)
    final_training_loss = regression_model.calculate_loss(
        regression_targets, 
        final_predictions
    )
    
    print(f"\nFinal Training Loss: {final_training_loss:.6f}")
    print("=" * 60)