# MNIST Neural Network Classifier

This project implements a simple feedforward neural network from scratch using NumPy to classify handwritten digits from the MNIST dataset.

## What is MNIST?

MNIST is a dataset of 70,000 handwritten digits (0-9), each represented as a 28x28 pixel grayscale image. It's a standard benchmark dataset in machine learning.

## Neural Network Fundamentals

For someone who knows how to code but is new to neural networks, here's how they work:

### The Core Concept

A neural network is essentially a mathematical function that transforms input data through a series of operations to produce output predictions. It learns to make these predictions by adjusting internal parameters based on example data.

### Network Architecture

Our specific implementation is a 2-layer neural network:

1. **Input Layer**: 784 neurons (28x28 pixels flattened)
2. **Hidden Layer**: 128 neurons 
3. **Output Layer**: 10 neurons (one for each digit 0-9)

### How Information Flows

1. **Forward Pass (Prediction)**:
   - Input values (pixels) are multiplied by weights and summed with biases
   - This sum is passed through an activation function (sigmoid)
   - The process repeats for each layer
   - The output layer produces 10 values representing the probability of each digit

2. **Backward Pass (Learning)**:
   - Calculate the error (difference between prediction and true label)
   - Propagate this error backward through the network
   - Adjust weights and biases to reduce the error
   - This process is called "backpropagation"

## Implementation Details

### Key Components

1. **Activation Function**: Sigmoid (maps any input to a value between 0 and 1)
2. **Loss Function**: Mean Squared Error (measures prediction error)
3. **Optimization**: Mini-batch Gradient Descent (updates weights in small batches)
4. **Weight Initialization**: Xavier/Glorot initialization (helps convergence)

### Code Breakdown

#### Data Preparation
- Load MNIST data using scikit-learn
- Normalize pixel values to [0,1] range
- Convert labels to one-hot encoding (e.g., 5 → [0,0,0,0,0,1,0,0,0,0])
- Split into training and test sets

#### Network Initialization
- Create weight matrices (W1, W2) with appropriate initialization
- Create bias vectors (b1, b2)

#### Training Loop
- For each epoch:
  - Shuffle training data
  - Process in mini-batches
  - Perform forward pass to get predictions
  - Calculate error
  - Perform backward pass to compute gradients
  - Update weights and biases
  - Track and report loss

#### Mathematical Operations

1. **Forward Pass**:
   - First layer: z1 = X·W1 + b1
   - First activation: a1 = sigmoid(z1)
   - Second layer: z2 = a1·W2 + b2
   - Output: a2 = sigmoid(z2)

2. **Backward Pass**:
   - Output error: error = y_true - a2
   - Output gradient: dz2 = error * sigmoid_derivative(a2)
   - Update W2: dW2 = a1ᵀ · dz2
   - Update b2: db2 = sum(dz2)
   - Hidden layer error: dz1 = (dz2 · W2ᵀ) * sigmoid_derivative(a1)
   - Update W1: dW1 = Xᵀ · dz1
   - Update b1: db1 = sum(dz1)

3. **Parameter Updates**:
   - W2 = W2 + learning_rate * dW2
   - b2 = b2 + learning_rate * db2
   - W1 = W1 + learning_rate * dW1
   - b1 = b1 + learning_rate * db1

## Hyperparameters

- **Learning Rate**: 0.01 (controls how quickly weights are updated)
- **Epochs**: 15 (number of complete passes through the training dataset)
- **Batch Size**: 128 (number of training examples processed before updating weights)
- **Hidden Layer Size**: 128 neurons

## Visualizations

The code generates a visualization of 10 test examples, showing the original image, the predicted digit, and the true label.

## Running the Code

To run this neural network:

```bash
python main.py
```

You'll see progress updates for each epoch and a final accuracy score. A visualization of sample predictions will be saved as 'predictions.png'.

## Extending the Project

Ideas for improvement:
- Add more hidden layers
- Try different activation functions (ReLU, tanh)
- Implement dropout for regularization
- Use cross-entropy loss instead of MSE
- Add momentum to the optimizer 