import os  # for file handling
import cv2  # computer vision
import numpy as np  # used for numpy arrays
import tensorflow as tf  # machine learning
import matplotlib.pyplot as plt  # for visualization

# Load the trained model, be sure to adjust to have the correct file name
model = tf.keras.models.load_model('improved_handwritten.keras')

# Reset the counter
prediction_counter = 1

# Define the folder containing test images
image_folder = "Images"

# Helper function to preprocess the image
def preprocess_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

    if img is None:
        return None  # Handle invalid or unreadable image files

    white_pixel_ratio = np.sum(img > 127) / img.size  # Check brightness

    if white_pixel_ratio > 0.5:
        img = cv2.bitwise_not(img)

    img = resize_with_padding(img, target_size=(28, 28))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Function to resize while keeping aspect ratio
def resize_with_padding(img, target_size=(28, 28)):
    old_size = img.shape[:2]
    ratio = float(target_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
    return img

# Loop through all images in the folder
for file_name in os.listdir(image_folder):
    if file_name.endswith((".png", ".jpg", ".jpeg")):
        file_path = os.path.join(image_folder, file_name)
        img = preprocess_image(file_path)
        if img is None:
            print(f"Skipping invalid image: {file_name}")
            continue
        # Make a prediction
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        print(f"{file_name}: This digit is probably a {predicted_digit}")
        # Display the image
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.title(f"Prediction: {predicted_digit}")
        plt.axis('off')
        plt.show()