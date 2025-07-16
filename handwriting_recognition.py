import os  # For file handling and directory operations
import cv2  # OpenCV for computer vision and image processing
import numpy as np  # NumPy for numerical operations and array handling
import tensorflow as tf  # TensorFlow for loading and using the trained model
import matplotlib.pyplot as plt  # Matplotlib for visualizing images and predictions

# Load the trained model from file
# The model was previously trained and saved by train_model.py
model = tf.keras.models.load_model('improved_handwritten.keras')

# Counter for tracking predictions (initialized but not currently used)
prediction_counter = 1

# Define the folder containing test images to be processed
image_folder = "Images"

# Helper function to preprocess the image for model input
def preprocess_image(file_path):

    # Read image in grayscale mode
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    # Return None if image couldn't be read
    if img is None:
        return None
    
    # Calculate ratio of white pixels to determine if inversion is needed
    # MNIST expects white digits on black background
    white_pixel_ratio = np.sum(img > 127) / img.size
    
    # If image has more white pixels than black (light background), invert it
    if white_pixel_ratio > 0.5:
        img = cv2.bitwise_not(img)
    
    # Resize image to 28x28 pixels (MNIST format) while maintaining aspect ratio
    img = resize_with_padding(img, target_size=(28, 28))
    
    # Normalize pixel values to range [0,1] (original range is [0,255])
    img = img / 255.0
    
    # Add batch dimension required by TensorFlow
    img = np.expand_dims(img, axis=0)
    
    return img

# Function to resize image while preserving aspect ratio and adding padding if needed
def resize_with_padding(img, target_size=(28, 28)):

    # Get original dimensions
    old_size = img.shape[:2]
    
    # Calculate scaling ratio based on the larger dimension
    ratio = float(target_size[0]) / max(old_size)
    
    # Calculate new size while maintaining aspect ratio
    new_size = tuple([int(x * ratio) for x in old_size])
    
    # Resize image using the calculated dimensions
    img = cv2.resize(img, (new_size[1], new_size[0]))
    
    # Calculate padding needed to reach target size
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    
    # Calculate padding for each side to center the image
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Add padding (black border) around the image
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
    
    return img

# Main processing loop, iterate through all images in the specified folder
print(f"Looking for images in the '{image_folder}' folder...")

# Loop through all files in the image folder
for file_name in os.listdir(image_folder):
    # Process only image files with supported extensions
    if file_name.endswith((".png", ".jpg", ".jpeg")):
        # Construct the full file path
        file_path = os.path.join(image_folder, file_name)
        
        # Preprocess the image for model input
        img = preprocess_image(file_path)
        
        # Skip invalid or unreadable images
        if img is None:
            print(f"Skipping invalid image: {file_name}")
            continue
            
        # Make a prediction using the trained model
        # The model returns an array of probabilities for each digit (0-9)
        prediction = model.predict(img, verbose=0)  # Suppress verbose output
        
        # Get the digit with highest probability
        # np.argmax returns the index (0-9) with the highest value
        predicted_digit = np.argmax(prediction)
        
        # Print the prediction result
        print(f"{file_name}: This digit is probably a {predicted_digit}")
        
        # Display the processed image with the prediction
        plt.figure(figsize=(4, 4))  # Set a consistent figure size
        plt.imshow(img[0], cmap=plt.cm.binary)  # Display in binary colormap (black and white)
        plt.title(f"Prediction: {predicted_digit}")
        plt.axis('off')  # Hide the axes for cleaner visualization
        plt.show()
        
print("Processing complete.")