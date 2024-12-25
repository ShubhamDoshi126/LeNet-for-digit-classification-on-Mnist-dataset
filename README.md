# LeNet-for-digit-classification-on-Mnist-dataset
This project demonstrates how to implement, train, and evaluate a Convolutional Neural Network (CNN) called LeNet for digit classification on the MNIST dataset. It includes functionalities for loading and preprocessing images, training the model, making predictions, and displaying results.

Files
module8.ipynb: Jupyter Notebook for creating, training, and saving the LeNet model using TensorFlow and Keras.
module8.py: Python script for loading an image, preprocessing it, and making predictions using the trained LeNet model.
le_net.py: Python script defining the LeNet CNN architecture and methods for training, saving, loading, and predicting.

Key Learnings
Convolutional Neural Network (CNN) Implementation
LeNet Architecture: Learn how to implement the LeNet CNN architecture using TensorFlow's Sequential API, including convolutional layers, pooling layers, and fully connected layers.
Layer Functions: Understand the role of each layer in the CNN and how they contribute to feature extraction and classification.
Model Compilation and Training
Model Compilation: Learn how to compile the CNN model with an optimizer, loss function, and evaluation metrics.
Training Process: Understand how to train the model on the MNIST dataset, including data preprocessing, fitting the model, and evaluating its performance.
Image Preprocessing for CNN
Preprocessing Steps: Learn how to preprocess images to match the input requirements of the CNN, including resizing, normalization, and reshaping.
Compatibility: Understand the importance of preprocessing steps in ensuring that the input data is compatible with the model.
Saving and Loading Models
Model Saving: Learn how to save a trained model to a file using TensorFlow's save method.
Model Loading: Understand how to load a saved model using the load method for future use and deployment.
Command-Line Interface for Predictions
CLI for Predictions: Learn how to create a command-line interface for loading and preprocessing images, making predictions, and displaying results.
Argument Handling: Understand how to handle command-line arguments and provide user-friendly error messages.
Visualization of Predictions
Prediction Visualization: Learn how to visualize the input image with prediction results using Matplotlib.
Interpretation: Understand the importance of visualizing predictions to interpret and validate the model's performance.

Usage

Training the LeNet Model
Run the following command in the Jupyter Notebook to create, train, and save the LeNet model:
from le_net import LeNet

# Create and train the model
lenet = LeNet(batch_size=128, epochs=10)
lenet.train()

# Save the trained model
lenet.save("doshi_cnn_model")

Loading and Predicting with the LeNet Model
Run the following command to load an image, preprocess it, and make predictions using the trained LeNet model:
python module8.py <image_filename> <expected_digit>
Example: python module8.py 0_0.png 0

Conclusion
This project provides a comprehensive understanding of implementing, training, and evaluating Convolutional Neural Networks (CNNs) for digit classification tasks using the LeNet architecture. It demonstrates the integration of various tools and libraries to create a complete workflow for image classification, including training, saving, loading, and making predictions with a user-friendly command-line interface.

Similar code found with 2 license types - View matches
Edit with Copilot

