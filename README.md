# GaTech - Deep Learning Course

This is the CS 7643 Deep Learning course from Georgia Tech. Please see the course contents [here](https://sites.cc.gatech.edu/classes/AY2023/cs7643_spring/).

## Overview

A comprehensive deep learning course covering neural network foundations from the ground up, progressing from basic building blocks to advanced model architectures and applications.

## Neural Network Foundations

- Forward and backward propagation (manual implementation with NumPy)
- Activation functions: ReLU, Sigmoid, Softmax
- Loss functions: Cross-Entropy Loss, Focal Loss
- Gradient computation via chain rule and backpropagation
- Weight initialization strategies
- Optimizers: Stochastic Gradient Descent (SGD)

## Model Architectures

- **Softmax Regression** — single-layer linear classifier
- **Multi-Layer Perceptron (MLP)** — two-layer fully connected network with non-linear activations
- **Convolutional Neural Networks (CNN)** — convolution, max pooling, and linear layers built from scratch
- **ResNet** — residual connections and skip connections for deeper networks (CIFAR-10)
- **Seq2Seq (Encoder-Decoder)** — RNN-based sequence-to-sequence model with LSTM
- **Transformer** — self-attention mechanism for sequence modeling
- **Multi-Input Model** — custom architecture combining Conv2D branches with multiple linear branches (final project)

## Assignments

| Assignment | Topics | Implementation |
|---|---|---|
| **Assignment 1** | Softmax regression, two-layer NN, SGD optimizer | NumPy from scratch |
| **Assignment 2 - Part 1** | Convolution, max pooling, linear layers, CNN classifier | NumPy from scratch |
| **Assignment 2 - Part 2** | Vanilla CNN, ResNet-32, Focal Loss, custom model | PyTorch |
| **Assignment 3** | Machine translation with LSTM, Seq2Seq, Transformer | PyTorch |
| **Final Project** | Multi-input sensor fusion model (acceleration, rotation, temperature, ToF) | PyTorch |

## Key Learnings

- Building neural networks from scratch deepens understanding of how gradients flow through layers
- Transitioning from NumPy to PyTorch highlights what frameworks abstract away (autograd, layer modules, optimizers)
- Implementing convolution and pooling operations manually before using `nn.Conv2d` and `nn.MaxPool2d`
- Understanding attention mechanisms and positional encoding in Transformers
- Designing custom architectures for multi-modal data fusion
