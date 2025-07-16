# Handwritten Digit Recognizer

A machine learning project that trains a neural network to recognize handwritten digits using the MNIST dataset. The model achieves high accuracy through a carefully designed architecture with multiple layers and dropout regularization. After training, the model can be used to predict digits from custom handwritten images.

## Features

- Builds and trains a deep neural network on the MNIST dataset
- Implements dropout layers and early stopping
- Displays training/validation loss curves and test accuracy
- Predicts digits from your own handwritten images
- Automatically processes images for optimal recognition

## Requirements

All dependencies are listed in the `requirements.txt` file. The main requirements are:
- Python 3.6+
- TensorFlow 2.5+
- NumPy
- Matplotlib
- OpenCV

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Borbez2/Handwritten-Digit-Recognizer.git
   cd Handwritten-Digit-Recognizer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the training script to build and train the neural network:

```bash
python3 train_model.py
```

This will:
- Load and preprocess the MNIST dataset
- Train the neural network
- Save the trained model as `improved_handwritten.keras`
- Display the training/validation loss graph

You can modify hyperparameters in the script to experiment with different configurations.

### Testing Custom Images

1. Place your handwritten digit images in the `Images` folder (PNG, JPG, or JPEG format)
2. Run the recognition script:
   ```bash
   python3 handwriting_recognition.py
   ```
3. The script will display each image with its predicted digit

Note: There are 350 sample test images already provided in the `Images` folder.

## Results

The model typically achieves 97-98%+ accuracy on the MNIST test set. Performance on custom images may vary depending on image quality and similarity to the MNIST dataset style.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
