# TSAI-S8
# Handwritten Digit Classifier

This is a deep neural network classiifier that uses covolution neural network with BN, LN and GroupNorm.

## MNIST Dataset

MNIST dataset contains gray scale images of size 28 * 28.
The train data has 60000 images and test has 10000 images.
Below is the sample data

## Architecture Diagram
The Architecture used for this implementation is of type squeeze and expand.

## Requirements
* matplotlib==3.7.1
* matplotlib-inline==0.1.6
* torch==2.0.1
* torchsummary==1.5.1
* torchvision==0.15.2
* tqdm==4.65.0

## Execution
To run the code, execute the S7_Assignment_Solution.ipynb file.
The model is set to execute on MPS/GPU/CPU
The training time on MPS is ~ 4 mins 

## Model Accuracy and Loss
The model consistenly reached and accuracy of 75.65% on test set.
The loss function used here was negative log liklihood loss.

## Model specs
* Total number of final trainable model parameters = 7654
* Total number of non-trainable model parameters = 0
* Model Size = 0.83 MB

## Authors

- [@mrinalmouza](https://github.com/mrinalmouza)