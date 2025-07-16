import tensorflow as tf  # Machine learning
from tensorflow.keras import Sequential  # Sequential model type
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input  # Neural network layers
from tensorflow.keras.callbacks import EarlyStopping  # Stops training when the model stops improving
import matplotlib.pyplot as plt  # Used for visualization

# Load the MNIST dataset (28x28 images of handwritten digits 0-9)
mnist = tf.keras.datasets.mnist

# Data will be split into testing and training data
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Returns 2 tuples with training and testing data

# Normalize the data so that the values are between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Create the neural network model
model = Sequential([
    Input(shape=(28, 28)),  # Input layer for 28x28 images
    Flatten(),  # Flattens 28x28 matrix into a 1D array of size 784
    Dense(128, activation='relu'),  # First hidden layer with 128 neurons
    Dropout(0.5),  # Dropout layer to prevent overfitting (randomly drops x% of neurons 30-60% results in most optimal results)
    Dense(128, activation='relu'),  # Second hidden layer with 128 neurons
    Dropout(0.5),  # Another dropout layer to prevent overfitting
    Dense(128, activation='relu'),  # Third hidden layer with 128 neurons
    Dropout(0.5),  # Another dropout layer to prevent overfitting
    Dense(10, activation='softmax')  # Output layer for 10 digits (0-9), using softmax for probabilities
])

# Compile the model
model.compile(optimizer='adam',  # Adam optimizer for adaptive learning rate
              loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification
              metrics=['accuracy'])  # Track accuracy during training

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss',  # Watch validation loss
                               patience=5,  # Stop training if no improvement after x epochs
                               restore_best_weights=True)  # Restore the best weights from training

# Train the actual model
history = model.fit(x_train, y_train,  # Train on the normalized training data
                    validation_split=0.1,  # Use 10% of training data for validation
                    epochs=50,  # Set a high max number of epochs
                    callbacks=[early_stopping])  # Use early stopping to stop automatically

# Save the trained model
model.save('improved_handwritten.keras')  # Save the trained model to a file

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(x_test, y_test)  # Test the model on unseen test data

# Output the loss and accuracy
print(f"Test Loss: {loss:.2f}")  # Print the final test loss
print(f"Test Accuracy: {accuracy * 100:.2f}%")  # Print the final test accuracy

# Plot training and validation loss, allows you to see if overfitting is happening
plt.plot(history.history['loss'], label='Training Loss')  # Plot the training loss over epochs
plt.plot(history.history['val_loss'], label='Validation Loss')  # Plot the validation loss over epochs
plt.xlabel('Epochs')  # Label the x-axis as "Epochs"
plt.ylabel('Loss')  # Label the y-axis as "Loss"
plt.legend()  # Show the legend to differentiate between training and validation loss
plt.title('Training vs Validation Loss')  # Add a title to the graph
plt.show()  # Display the graph