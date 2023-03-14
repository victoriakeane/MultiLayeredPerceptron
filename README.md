# MultiLayeredPerceptron
Simple Neural Network built as part of a college project. Report of results regarding processed data and predicted outcomes is available in report. Raw accuracy ratings available in "varied results" folders. This Perceptron takes numerical input data and produces a prediction.

To complete this project I implemented formulas for a hyperbolic tangent and logistic sigmoid as the activation functions. The activation functions determines whether input nodes can be "activated" and return a new output, or in other terms - are integral to deriving a prediction from. Nodes are then checked for errors through backpropogation, which passes back the loss through the network in order to readjust weights. Weights are values attached to each input and hidden node that represent their importance as features for predicting the final output 

## Installation

The only dependency for this project is having numpy installed on your machine. To see the MLP at work simply run either the Q1 or Q3 files, which will produce .txt files that document training and testing results as epochs are incremented. 

As a simple explanation of functionality, the Q1 file contains sets of numeric XOR tables represented as arrays, when passed through the MLP, the correct XOR outcome should be predicted for each input.
