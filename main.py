import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# ================ HELPER FUNCTIONS ================

# Activation functions
def sigmoid(x):
    # Sigmoid function maps any input to a value between 0 and 1
    # Clip values to avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Derivative of sigmoid function used during backpropagation
    # Note: This assumes x is already sigmoid(input)
    return x * (1 - x)

def one_hot(y, num_classes=10):
    # Convert digit labels (0-9) to one-hot encoded vectors
    # Example: 5 -> [0,0,0,0,0,1,0,0,0,0]
    oh = np.zeros((y.size, num_classes))
    # Set 1s at specific positions using array indexing:
    # np.arange(y.size) creates row indices [0,1,2,...,len(y)-1]
    # y.astype(int) provides the column indices based on label values
    # Together they form coordinate pairs (row,col) where 1s should be placed
    oh[np.arange(y.size), y.astype(int)] = 1
    return oh

# ================ DATA PREPARATION ================

# Load and preprocess MNIST data
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')

# Convert pandas to numpy
X = X.to_numpy() # This holds the pixels, 2 dimensional array, 784 pixels
y = y.to_numpy().astype(int) # This holds the labels

# Normalize pixel values from [0,255] to [0,1]
# This helps with training stability and convergence
X = X / 255.0 #So the pixel values are between 0 and 1

# Split into train and test sets (60,000 training, 10,000 test)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)
#x_train holds the images for training
#x_test holds the images for testing
#y_train holds the labels for training, labels are the actual digits
#y_test holds the labels for testing, labels are the actual digits

# Convert labels to one-hot encoding for training
y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

# ================ NETWORK CONFIGURATION ================

# Set hyperparameters
input_size = 784  # 28x28 pixel images flattened to 784 features
hidden_size = 128  # Number of neurons in hidden layer
output_size = 10  # 10 possible output classes (digits 0-9)
learning_rate = 0.01  # Controls how quickly weights are updated
epochs = 15  # Number of complete passes through the training data
batch_size = 128  # Number of samples processed before weights are updated

# ================ NETWORK INITIALIZATION ================

# Initialize weights and biases with Xavier/Glorot initialization
# This helps prevent vanishing/exploding gradients
np.random.seed(42)  # For reproducibility
# Weight matrices are initialized to small random values scaled by input dimensions
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0/input_size)  # Input → Hidden
b1 = np.zeros((1, hidden_size))  # Hidden layer bias
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0/hidden_size)  # Hidden → Output
b2 = np.zeros((1, output_size))  # Output layer bias

# Calculate number of batches
n_batches = len(x_train) // batch_size

# ================ TRAINING LOOP ================

for epoch in range(epochs):
    # Shuffle the training data at the start of each epoch
    # This helps prevent the network from learning the order of examples
    indices = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[indices]
    y_train_oh_shuffled = y_train_oh[indices]
    
    epoch_loss = 0
    
    # Mini-batch training
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(x_train))
        
        # Get the current mini-batch
        x_batch = x_train_shuffled[start:end]
        y_batch = y_train_oh_shuffled[start:end]
        
        # ===== FORWARD PASS =====
        # 1. Compute weighted sum at hidden layer
        z1 = np.dot(x_batch, W1) + b1  # Input × weights + bias
        # 2. Apply activation function to get hidden layer output
        a1 = sigmoid(z1)
        # 3. Compute weighted sum at output layer
        z2 = np.dot(a1, W2) + b2  # Hidden × weights + bias
        # 4. Apply activation function to get network output
        a2 = sigmoid(z2)  # Final prediction (probabilities for each digit)

        # ===== COMPUTE ERROR =====
        # Difference between true labels and predictions
        error = y_batch - a2
        # Mean squared error loss
        batch_loss = np.mean(np.square(error))
        epoch_loss += batch_loss

        # ===== BACKWARD PASS (BACKPROPAGATION) =====
        # 1. Calculate output layer gradient
        dz2 = error * sigmoid_derivative(a2)
        # 2. Calculate hidden-to-output weight gradient
        dW2 = np.dot(a1.T, dz2)
        # 3. Calculate output bias gradient
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # 4. Calculate hidden layer gradient (error propagated back)
        dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
        # 5. Calculate input-to-hidden weight gradient
        dW1 = np.dot(x_batch.T, dz1)
        # 6. Calculate hidden bias gradient
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # ===== UPDATE WEIGHTS AND BIASES =====
        # Update weights and biases using gradient descent
        W2 += learning_rate * dW2  # Update hidden-to-output weights
        b2 += learning_rate * db2  # Update output biases
        W1 += learning_rate * dW1  # Update input-to-hidden weights
        b1 += learning_rate * db1  # Update hidden biases
    
    # Print training progress after each epoch
    epoch_loss /= n_batches
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# ================ EVALUATION ================

# Prediction function for making inference
def predict(x):
    # Forward pass for prediction
    a1 = sigmoid(np.dot(x, W1) + b1)
    a2 = sigmoid(np.dot(a1, W2) + b2)
    # Return the index of the highest probability (predicted digit)
    return np.argmax(a2, axis=1)

# Evaluate accuracy on test set
predictions = predict(x_test)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# ================ VISUALIZATION ================

# Visualize some predictions on test data
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    # Reshape 1D array back to 2D image (28×28)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    pred = predict(x_test[i].reshape(1, -1))[0]
    plt.title(f"Pred: {pred}, True: {y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('predictions.png')
print("Sample predictions saved to 'predictions.png'")