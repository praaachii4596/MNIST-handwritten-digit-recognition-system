# MNIST Handwritten Digit Recognition – Google Colab

This repository contains a complete workflow for building and evaluating a neural network that recognizes handwritten digits using the MNIST dataset. The code is implemented in Python using Keras/TensorFlow, Matplotlib, NumPy, OpenCV, and runs in Google Colab.

## Features

- **Data Loading:** Downloads and splits the classic MNIST dataset (60,000 training, 10,000 test images, 28x28 grayscale).
- **Exploration:** Visualizes digits using Matplotlib and verifies label correspondence.
- **Preprocessing:**  
  - Normalizes images to 0-1 range.  
  - Converts digit labels to one-hot encoding for training.
- **Model Building:**  
  - Sequential neural network:  
    - `Flatten` input layer (28x28 to 784).  
    - `Dense` hidden layer with 128 ReLU activations.  
    - `Dense` output layer with 10 softmax units.
- **Training:**  
  - Compiles with Adam optimizer and categorical cross-entropy.  
  - Trains for 10 epochs with batch size 32, reports accuracy and loss curves.
- **Evaluation:**  
  - Gives model accuracy on the test set (typically ~98%).
- **Prediction System:**  
  - Predicts digit class for test images and shows softmax output.  
  - Converts probabilities to integer labels (`argmax`).
- **Custom Image Prediction:**  
  - Accepts external images (e.g., JPG, JPEG, PNG), displays them, converts to grayscale, resizes to 28x28, inverts colors if needed.
  - Runs custom images through the model and prints predicted digit.

## Usage

### Open the Notebook

- Launch directly in Google Colab or upload to your own Colab environment.
- [Open in Colab](https://colab.research.google.com/...)  

### Dependencies

- No installs required in Colab, but uses:  
  - `numpy`  
  - `matplotlib`  
  - `tensorflow` (Keras)  
  - `seaborn`  
  - `opencv-python`

### Walkthrough

1. **Import Libraries & Load Data**
```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

2. **Visualize & Explore Data**
- Use `plt.imshow(x_train[img_index])` and `print(y_train[img_index])`.
  
3. **Preprocess Data**
```python
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
```

4. **Build Model**
```python
model = Sequential([
Flatten(input_shape=(28, 28)),
Dense(128, activation='relu'),
Dense(10, activation='softmax')
])
```

5. **Train Model**
```python
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
```

6. **Evaluate Model**
```python
loss, accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', accuracy)
```

7. **Make Predictions**
- Predict test digits and print both softmax output and argmax label.
- For custom images:  
  - Read using OpenCV, convert to grayscale, resize, invert (only if digit is black on white background) and scale, reshape, and predict.
```python
img = cv2.imread('/content/download.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_resized = cv2.resize(gray, (28, 28))
img_resized = 255 - img_resized
img_resized = img_resized / 255.0
img_reshaped = np.reshape(img_resized, (1, 28, 28))
input_pred = model.predict(img_reshaped)
print(np.argmax(input_pred))
```

## Notes

- All code and visualizations are annotated and well-structured for study and adaptation.
- The system is fully interactive and works for both MNIST and user-uploaded digit images.

## Example Prediction 
<img width="398" height="418" alt="image" src="https://github.com/user-attachments/assets/43d939f5-134c-4c2c-a8fe-eaf3cf108252" />
<img width="398" height="418" alt="image" src="https://github.com/user-attachments/assets/90a7288b-1525-4603-8ec7-a04930b6ba65" />


**Happy digit recognizing!**






