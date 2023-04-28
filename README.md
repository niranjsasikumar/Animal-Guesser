# Animal Guesser

An image classification program powered by the PyTorch deep learning framework that predicts which animal is in a given image.

## build_model.py

Uses a pretrained Dense Convolutional Network (DenseNet) model. The classifier of the model is a sequential neural network consisting of three fully-connected layers with ReLU activation functions, and a log softmax output layer. The training data consists of images of 40 different types of animals. The model is trained using stochastic gradient descent with the negative log-likelihood loss.

## model.pth

The output of build_model.py. Contains a trained model as specified above.

## animal_guesser.py

Loads the model from model.pth. Prompts the user to choose whether to input the image path as a URL or a local file. Checks if the provided path is valid and if it contains an image. Preprocesses the image and returns the image as a PyTorch tensor. Runs the tensor through the model to obtain the predicted class probabilities. Returns a string with the predicted animal name and the probability of the prediction.
