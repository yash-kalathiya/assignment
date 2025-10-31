## Assignment Overview

This repository contains my solutions for the neural networks homework assignment.

## Files

- `task1.py` - Explicit formula derivation for neural network
- `task2.py` - Neural network implementation with training
- `README.md` - This documentation

## Task 1: Explicit Formula for f

![Neural Network Diagram](network_diagram.png)

The given network takes the input vector:
```
x = [1, x₁, x₂]ᵀ
```

and uses several threshold (step) units defined as:
```
H(z) = 1, if z ≥ 0
H(z) = 0, if z < 0
```

### Step 1 – First Hidden Layer
Let the first two neurons compute:
```
h₁ = H(w₁ᵀx)
h₂ = H(w₂ᵀx)
```
These are the basic activations from the input layer.

### Step 2 – Second Hidden Layer
From the diagram, both second-layer units have a bias of –1.5.
- The first one connects with +1 from h₁ and –1 from h₂.
- The second one connects with –1 from h₁ and +1 from h₂.

Therefore:
```
g₁ = H(-1.5 + h₁ - h₂)
g₂ = H(-1.5 - h₁ + h₂)
```

### Step 3 – Output Layer
The final node has bias +1.5 and takes inputs from both g₁ and g₂ with weight 1:
```
f = H(1.5 + g₁ + g₂)
```

### Step 4 – Combined Formula
Substituting all the terms, the final formula becomes:
```
f(x) = H(1.5 + H(-1.5 + H(w₁ᵀx) - H(w₂ᵀx)) + H(-1.5 - H(w₁ᵀx) + H(w₂ᵀx)))
```

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