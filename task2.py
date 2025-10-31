import numpy as np


class TwoLayerPerceptron:
    
    def __init__(self, hidden_nodes=5, alpha=0.01, seed_value=42):
        np.random.seed(seed_value)
        
        self.hidden_nodes = hidden_nodes
        self.alpha = alpha
        
        input_dimension = 3
        
        fan_input_first = input_dimension
        fan_output_first = self.hidden_nodes
        std_dev_first = np.sqrt(2.0 / (fan_input_first + fan_output_first))
        self.weight_matrix_layer1 = np.random.normal(
            loc=0.0, 
            scale=std_dev_first, 
            size=(input_dimension, self.hidden_nodes)
        )
        
        fan_input_second = self.hidden_nodes + 1
        fan_output_second = 1
        std_dev_second = np.sqrt(2.0 / (fan_input_second + fan_output_second))
        self.weight_matrix_layer2 = np.random.normal(
            loc=0.0,
            scale=std_dev_second,
            size=(self.hidden_nodes + 1, 1)
        )
        
        self.hidden_pre_activation = None
        self.hidden_post_activation = None
        self.output_pre_activation = None
        self.output_post_activation = None
        self.input_with_bias = None
        
    def forward_pass(self, input_data):
        num_samples = input_data.shape[0]
        
        bias_column = np.ones((num_samples, 1))
        self.input_with_bias = np.column_stack([bias_column, input_data])
        
        self.hidden_pre_activation = self.input_with_bias @ self.weight_matrix_layer1
        self.hidden_post_activation = np.tanh(self.hidden_pre_activation)
        
        hidden_with_bias = np.column_stack([bias_column, self.hidden_post_activation])
        self.output_pre_activation = hidden_with_bias @ self.weight_matrix_layer2
        self.output_post_activation = self.output_pre_activation
        
        return self.output_post_activation
    
    def calculate_loss(self, target_values, predicted_values):
        squared_errors = (target_values - predicted_values) ** 2
        return 0.5 * np.mean(squared_errors)
    
    def backward_pass(self, target_values):
        num_samples = target_values.shape[0]
        
        output_error_gradient = (self.output_post_activation - target_values) / num_samples
        
        bias_column = np.ones((num_samples, 1))
        hidden_with_bias = np.column_stack([bias_column, self.hidden_post_activation])
        gradient_weight_layer2 = hidden_with_bias.T @ output_error_gradient
        
        hidden_error = output_error_gradient @ self.weight_matrix_layer2[1:].T
        
        tanh_derivative = 1 - np.square(np.tanh(self.hidden_pre_activation))
        hidden_delta = hidden_error * tanh_derivative
        
        gradient_weight_layer1 = self.input_with_bias.T @ hidden_delta
        
        return gradient_weight_layer1, gradient_weight_layer2
    
    def apply_gradient_update(self, gradient_layer1, gradient_layer2):
        self.weight_matrix_layer1 -= self.alpha * gradient_layer1
        self.weight_matrix_layer2 -= self.alpha * gradient_layer2
    
    def perform_training_iteration(self, input_data, target_values):
        predictions = self.forward_pass(input_data)
        iteration_loss = self.calculate_loss(target_values, predictions)
        
        grad_layer1, grad_layer2 = self.backward_pass(target_values)
        self.apply_gradient_update(grad_layer1, grad_layer2)
        
        return iteration_loss
    
    def generate_predictions(self, input_data):
        return self.forward_pass(input_data)


def execute_training_procedure(feature_matrix, target_vector, hidden_layer_size=5, 
                                learning_rate_param=0.01, training_epochs=1000, 
                                mini_batch_size=None, show_progress=True):
    
    trained_model = TwoLayerPerceptron(
        hidden_nodes=hidden_layer_size, 
        alpha=learning_rate_param
    )
    
    loss_trajectory = []
    total_samples = feature_matrix.shape[0]
    
    if mini_batch_size is None:
        mini_batch_size = total_samples
    
    for epoch_idx in range(training_epochs):
        epoch_loss_values = []
        
        if mini_batch_size < total_samples:
            shuffle_indices = np.random.permutation(total_samples)
            features_shuffled = feature_matrix[shuffle_indices]
            targets_shuffled = target_vector[shuffle_indices]
        else:
            features_shuffled = feature_matrix
            targets_shuffled = target_vector
        
        for batch_start in range(0, total_samples, mini_batch_size):
            batch_end = min(batch_start + mini_batch_size, total_samples)
            
            feature_batch = features_shuffled[batch_start:batch_end]
            target_batch = targets_shuffled[batch_start:batch_end]
            
            batch_loss_value = trained_model.perform_training_iteration(
                feature_batch, 
                target_batch
            )
            epoch_loss_values.append(batch_loss_value)
        
        mean_epoch_loss = np.mean(epoch_loss_values)
        loss_trajectory.append(mean_epoch_loss)
        
        if show_progress and (epoch_idx + 1) % (training_epochs // 10) == 0:
            print(f"Epoch {epoch_idx + 1}/{training_epochs}, Loss: {mean_epoch_loss:.6f}")
    
    return trained_model, loss_trajectory


if __name__ == "__main__":
    print("=" * 60)
    print("Task 2: Two-Layer Neural Network Implementation")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("Experiment 1: Circular Region Classification")
    print("=" * 60)
    
    np.random.seed(123)
    num_circle_samples = 100
    
    circle_features = np.random.rand(num_circle_samples, 2)
    
    center_x, center_y = 0.5, 0.5
    radius_threshold = 0.35
    
    distances = np.sqrt(
        (circle_features[:, 0] - center_x)**2 + 
        (circle_features[:, 1] - center_y)**2
    )
    
    circle_targets = (distances <= radius_threshold).astype(float).reshape(-1, 1)
    
    circle_model, circle_loss_history = execute_training_procedure(
        feature_matrix=circle_features,
        target_vector=circle_targets,
        hidden_layer_size=8,
        learning_rate_param=0.1,
        training_epochs=1500,
        show_progress=False
    )
    
    test_points = np.array([
        [0.5, 0.5],
        [0.3, 0.5],
        [0.1, 0.1],
        [0.9, 0.9],
        [0.5, 0.7],
        [0.2, 0.2],
    ])
    
    test_distances = np.sqrt(
        (test_points[:, 0] - center_x)**2 + 
        (test_points[:, 1] - center_y)**2
    )
    test_targets = (test_distances <= radius_threshold).astype(float)
    circle_predictions = circle_model.generate_predictions(test_points)
    
    print("\nCircular Classification Results (Test Points):")
    print("-" * 75)
    print(f"{'Point (x, y)':<20} {'Distance':<15} {'Expected':<15} {'Predicted':<15}")
    print("-" * 75)
    for sample_idx in range(len(test_points)):
        point_str = f"({test_points[sample_idx][0]:.2f}, {test_points[sample_idx][1]:.2f})"
        dist_str = f"{test_distances[sample_idx]:.4f}"
        expected_val = f"{test_targets[sample_idx]:.0f}"
        predicted_val = f"{circle_predictions[sample_idx][0]:.4f}"
        print(f"{point_str:<20} {dist_str:<15} {expected_val:<15} {predicted_val:<15}")
    
    train_predictions = circle_model.generate_predictions(circle_features)
    train_accuracy = np.mean((train_predictions > 0.5) == (circle_targets > 0.5))
    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
    
    print("\n" + "=" * 60)
    print("Experiment 2: Nonlinear Regression (Stochastic GD)")
    print("=" * 60)
    
    np.random.seed(42)
    num_regression_samples = 400
    regression_features = np.random.randn(num_regression_samples, 2)
    
    regression_targets = (
        np.sin(regression_features[:, 0]) + 
        np.cos(regression_features[:, 1]) + 
        0.1 * np.random.randn(num_regression_samples)
    ).reshape(-1, 1)
    
    regression_model, regression_loss_history = execute_training_procedure(
        feature_matrix=regression_features,
        target_vector=regression_targets,
        hidden_layer_size=10,
        training_epochs=1000,
        mini_batch_size=1
    )
    
    final_predictions = regression_model.generate_predictions(regression_features)
    final_training_loss = regression_model.calculate_loss(
        regression_targets, 
        final_predictions
    )
    
    print(f"\nFinal Training Loss: {final_training_loss:.6f}")
    print("=" * 60)