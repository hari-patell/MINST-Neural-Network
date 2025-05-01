import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Activation functions
def sigmoid(x):
    # Clip values to avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def one_hot(y, num_classes=10):
    oh = np.zeros((y.size, num_classes))
    oh[np.arange(y.size), y.astype(int)] = 1
    return oh

# Load and preprocess MNIST data
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')

# Convert pandas to numpy
X = X.to_numpy()
y = y.to_numpy().astype(int)

# Normalize pixel values
X = X / 255.0

# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42)

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

# Set hyperparameters
input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.01  # Reduced learning rate
epochs = 15  # Increased epochs

# Initialize weights and biases with better initialization
np.random.seed(42)  # For reproducibility
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0/input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0/hidden_size)
b2 = np.zeros((1, output_size))

# Use mini-batches for better training
batch_size = 128
n_batches = len(x_train) // batch_size

# Training loop
for epoch in range(epochs):
    # Shuffle the training data
    indices = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[indices]
    y_train_oh_shuffled = y_train_oh[indices]
    
    epoch_loss = 0
    
    # Mini-batch training
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(x_train))
        
        # Get the mini-batch
        x_batch = x_train_shuffled[start:end]
        y_batch = y_train_oh_shuffled[start:end]
        
        # Forward pass
        z1 = np.dot(x_batch, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Compute error
        error = y_batch - a2
        batch_loss = np.mean(np.square(error))
        epoch_loss += batch_loss

        # Backpropagation
        dz2 = error * sigmoid_derivative(a2)
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
        dW1 = np.dot(x_batch.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights and biases
        W2 += learning_rate * dW2
        b2 += learning_rate * db2
        W1 += learning_rate * dW1
        b1 += learning_rate * db1
    
    # Print training progress
    epoch_loss /= n_batches
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# Prediction
def predict(x):
    a1 = sigmoid(np.dot(x, W1) + b1)
    a2 = sigmoid(np.dot(a1, W2) + b2)
    return np.argmax(a2, axis=1)

# Evaluate accuracy
predictions = predict(x_test)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Visualize some predictions
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    pred = predict(x_test[i].reshape(1, -1))[0]
    plt.title(f"Pred: {pred}, True: {y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('predictions.png')
print("Sample predictions saved to 'predictions.png'")