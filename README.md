## MicroLM
This project is a character-level language model built with PyTorch. I created it as a study exercise for an upcoming university exam to better understand the internal mechanics of neural networks, specifically backpropagation and embedding layers.

It serves as a refined, more modular version of my previous project, NanoLM. While NanoLM was a first attempt using raw tensors, MicroLM organizes the logic into proper nn.Module classes and separates training from inference.

## Overview
The model is a simple Multilayer Perceptron (MLP) trained on the TinyStories dataset. It takes a fixed window of characters as input, creates embeddings for them, flattens the result, and passes it through a hidden layer to predict the next character in the sequence.

## Architecture:

Embedding Layer: Converts character indices into dense vectors.

Flattening: Concatenates the embeddings of the context window.

Hidden Layer: Linear transformation followed by a ReLU activation.

Output Layer: Linear transformation projecting to the vocabulary size.
