import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Common practice for splitting
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# --- 1. Data Loading and Preparation ---
# !!! REPLACE THIS SECTION WITH YOUR ACTUAL DATA LOADING !!!
# Example using dummy data: Assume you have features 'X' and labels 'y'
# X should be a 2D numpy array (samples, features)
# y should be a 1D numpy array (samples,)
print("Loading and preparing data...")
# Generate some dummy data for demonstration
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=13, n_informative=8,
                           n_redundant=2, n_clusters_per_class=2, random_state=42)

# Ensure y is suitable for binary conversion (e.g., 0 and 1, or 0 and >0)
# If your labels are already 0 and 1, you might skip the conversion step below.

# Split data into training and testing sets
# test_size=0.2 means 20% of data is used for testing
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

# Convert into binary classification problem (if necessary)
# This assumes labels > 0 indicate the positive class (e.g., heart disease present)
# Adjust this logic based on your specific label encoding.
y_train_binary = y_train.copy()
y_test_binary = y_test.copy()
# Example: If y has multiple positive classes (1, 2, 3...) map them all to 1
y_train_binary[y_train_binary > 0] = 1
y_test_binary[y_test_binary > 0] = 1
# If y was already 0/1, this step might not change anything, which is fine.

print("Unique binary labels in y_train_binary:", np.unique(y_train_binary))
print("Unique binary labels in y_test_binary:", np.unique(y_test_binary))


# --- 2. Define the Keras Model ---
def create_binary_model(input_shape: int) -> Sequential:
    """
    Creates a Sequential Keras model for binary classification.

    Args:
        input_shape: The number of features in the input data.

    Returns:
        A compiled Keras Sequential model.
    """
    model = Sequential(name="Binary_Classification_Model")

    # Input Layer + First Hidden Layer
    model.add(Dense(16,                     # Number of neurons in the layer
                    input_shape=(input_shape,), # Shape of the input (number of features)
                    kernel_initializer='glorot_uniform', # How to initialize weights (default='glorot_uniform')
                    kernel_regularizer=regularizers.l2(0.001), # L2 regularization (prevents overfitting)
                    activation='relu'))    # Activation function (Rectified Linear Unit)
    model.add(Dropout(0.25)) # Dropout regularization (randomly sets inputs to 0)

    # Second Hidden Layer
    model.add(Dense(8,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(0.001),
                    activation='relu'))
    model.add(Dropout(0.25))

    # Output Layer
    model.add(Dense(1,                      # Single neuron for binary output
                    activation='sigmoid'))  # Sigmoid activation for probability (0 to 1)

    # Compile the model
    # Adam optimizer is a popular choice for gradient descent
    adam_optimizer = Adam(learning_rate=0.001) # Define the optimizer
    model.compile(loss='binary_crossentropy', # Loss function suitable for binary classification
                  optimizer=adam_optimizer,   # Optimizer to use for training
                  metrics=['accuracy'])       # Metric(s) to evaluate during training/testing
    return model

# Get the number of features from the training data
num_features = X_train.shape[1]

# Create the model instance
binary_model = create_binary_model(num_features)

# Print a summary of the model's layers and parameters
print("\nModel Summary:")
binary_model.summary()


# --- 3. Train the Model ---
print("\nTraining the model...")
# epochs: Number of times the entire training dataset is passed through the model
# batch_size: Number of samples processed before the model's weights are updated
# validation_data: Data used to evaluate the model's performance after each epoch (doesn't affect training)
history = binary_model.fit(X_train,
                           y_train_binary,
                           validation_data=(X_test, y_test_binary),
                           epochs=50,
                           batch_size=10,
                           verbose=1) # Set verbose=1 or 2 to see progress per epoch, 0 for silent


# --- 4. Visualize Training History ---
print("\nPlotting training history...")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss (Binary Crossentropy)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()


# --- 5. Evaluate the Model ---
print("\nEvaluating the model on the test set...")

# Get predictions (probabilities) from the model
# predict returns probabilities because of the sigmoid activation in the last layer
y_pred_proba = binary_model.predict(X_test)

# Convert probabilities to binary predictions (0 or 1)
# Threshold is typically 0.5 for sigmoid output
y_pred_binary = np.round(y_pred_proba).astype(int).flatten() # Use flatten() to make it 1D

# Calculate and print metrics
accuracy = accuracy_score(y_test_binary, y_pred_binary)
report = classification_report(y_test_binary, y_pred_binary)

print('\n--- Results for Binary Model ---')
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(report)

print("\nScript finished.")