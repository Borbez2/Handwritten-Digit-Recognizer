# Simple-Neural-Network-MNIST

This project trains a neural network to recognize handwritten digits using the MNIST dataset and then tests its performance. 
It also allows you to test the model on your custom images.

## Features:

Trains a neural network with multiple layers and dropout to prevent overfitting.
Implements early stopping to optimize training duration.
Visualizes the model with test accuracy and plots training vs. validation loss.
Predicts handwritten digits from custom images.

## Setup:

```
pip install tensorflow numpy matplotlib opencv-python
```
Clone this repository and go to the project folder

## Train the model:

Run train_model.py to train and save the neural network as improved_handwritten.keras.
You can tweak the epochs and dropout rate in the script to see which will produce better results. 

## To test custom images:

Place your images in a folder named Images.
Run handwriting_recognition.py to predict digits from the images.
There are 350 test sample images already provided in the Images folder.

## Outputs:

Training accuracy and validation loss graph.
Predicted digits for custom images displayed with the input image.
