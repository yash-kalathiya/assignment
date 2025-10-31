## Assignment Overview

This repository contains my solutions for the neural networks homework assignment.

## Files

- `task2.py` - Neural network implementation with training
- `README.md` - This documentation

## Task 2: Neural Network Implementation

Built a complete 2-layer neural network with:
- **Input:** 2 units (plus bias)
- **Hidden:** Configurable number of units with tanh activation
- **Output:** 1 unit with linear activation
- **Training:** Gradient descent with squared error loss

### Key Features
- Forward and backward propagation
- Batch and mini-batch training support
- Works on both XOR and regression problems

## Running the Code

```bash
# Make sure you have numpy and matplotlib installed
pip install numpy matplotlib

# Run Task 2
python task2.py
```

## Results

**Task 2:** 
- Solves XOR problem with near-perfect accuracy (< 0.0001 error)
- Achieves good performance on regression tasks
- Demonstrates stable training convergence

## Implementation Notes

I used standard neural network techniques like:
- Proper weight initialization 
- Matrix operations for efficiency
- Tanh activation for hidden layers
- Mean squared error loss function
- Gradient descent optimization

The implementation follows the assignment specifications closely while ensuring numerical stability and good performance.