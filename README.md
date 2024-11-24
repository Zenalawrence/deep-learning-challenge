# deep-learning-challenge


## Overview of the Analysis

The purpose of this analysis was to develop a deep learning model to classify applications in the Alphabet Soup dataset. The objective was to predict whether an application would be successful based on  application type and classification. This analysis aimed to leverage neural network capabilities to identify patterns and improve the classification accuracy compared to traditional methods.


## Technical Skills
Machine learning
Deep learning
Neural networks
Tensorflow
Pandas
Python

## Data Processing

 - Target Variables:  "IS_SUCCESSFUL" was a binary variable used as the target variable.  This determined whether an application was successful or not.

 - Feature Variables: 
    - APPLICATION_TYPE
    - AFFILIATION
    - CLASSIFICATION
    - USE_CASE
    - ORGANIZATION
    - STATUS
    - INCOME_AMT
    - SPECIAL_CONSIDERATIONS
    - ASK_AMT

 - Removed Variables: 
    - EIN
    - NAME

The final data set is shown below:
![Alphabet Soup Dataset.](https://github.com/Zenalawrence/xxx)

## Compiling, Training, and Evaluating the Model

### Attempt 1

The first model consisted of 2 hidden layers with 80 and 30 neurons respectively with 100 epochs.  The input later was the length of the x_trained_scaled, and the subsequent layers being halved each layer.  The activation function used was ReLU because of the non-linear nature of the data.  The output layer utilized the Sigmoid activation function to handle the binary classification problem.
![Model Training.](https://github.com/Zenalawrence/xxx)

This model resulted in an accuracy score: 0.7308 (73.1%) and loss: 0.5591 (55.8%).  

### Attempt 2

I increased the complexity to optimize this model. An additional hidden layer was added to allow the model to learn more complex representations.  This additional layer, was halved from the previous layer and kept 100 epochs.  The results were not that different from the initial attempt. 

![Model Training2.](https://github.com/Zenalawrence/xxx)

This model resulted in an accuracy score: 0.7292 (72.9%) and loss: 0.5671 (56.7%).

### Attempt 3

The complexity was increased further to allow for deeper learning by adding more neurons to the model. The three Hidden layers contained, 150, 75 and 25 neurons respectively with 100 epochs. The results remained consistent to previous attempts.

![Model Training2.](https://github.com/Zenalawrence/xxx)

This model resulted in an accuracy score: 0.7278 (72.8%) and loss: 0.5869 (58.7%).

### Attempt 4

In the final attempt, I performed hyperparameter tuning.  This created multiple model settings by creating variations in neuron numbers, layer counts, and activation functions.  The tuning was able to train with either ReLU and tanh activation functions for the hidden layers.  There was between 1 and 7 additional hidden layers, each with a varying number of units from 1 to 128, increasing in steps of 5.


This was the best resulting model, with an accuracy score: 0.7363 (73.6%) and loss: 0.5496 (54.9%).

## Summary

The final tuning process yielded the best results, suggesting that varying the architecture and activation functions had a positive impact on model accuracy.  To experiment futher,  exploring more activation functions and advanced architectures such as convolutional neural networks (CNNs) or recurrent neural networks (RNNs) may improve the accuracy.