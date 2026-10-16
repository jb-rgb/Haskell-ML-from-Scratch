# Haskell ML From Scratch

k-NN and Feedforward Neural Network classifiers for the MNIST dataset, written in pure Haskell from the ground up without any external Machine Learning libraries.

## Overview
This project implements mathematical models for pattern recognition and digit classification using the MNIST dataset. By strictly avoiding heavy Machine Learning libraries, this engine serves as a "white-box" study of the underlying linear algebra, calculus, and functional data transformations that power modern AI architectures. 

## Core Features
- **Pure Functional Linear Algebra:** Core operations (`dotProduct`, `matrixVectorMultiply`, `outerProduct`, `transposeMatrix`) built strictly using higher-order functions (`map`, `foldl`, `zipWith`).
- **k-Nearest Neighbors (k-NN):** A lazy learning classifier using pure Euclidean distance metrics and algorithmic mode selection.
- **Feedforward Neural Network:** A complete multi-layer perceptron architecture featuring hidden layers, ReLU activation, and Softmax output.
- **Backpropagation Engine:** Pure functional training loop using Stochastic Gradient Descent (SGD) with custom mathematical derivatives to calculate gradients and update weights.
- **Deterministic Pseudo-Random Initialization:** Custom linear congruential generator for symmetrical-breaking weight initialization.
- **Architectural Comparison:** Built-in execution flow that demonstrates the inference-time tradeoffs between Eager Learning (Neural Networks: ~96% accuracy, instant inference) and Lazy Learning (k-NN: high accuracy, computationally heavy inference).

## Tech Stack
- **Language:** Haskell (GHC 9.10+)
- **Build System:** Cabal
- **Paradigm:** Pure Functional Programming

## Getting Started

### 1. Prerequisites
Ensure you have `ghc` and `cabal-install` available on your system.

### 2. Dataset Preparation
This project uses the CSV version of the MNIST dataset to avoid complex binary parsing.
1. Download the "MNIST in CSV" dataset (e.g., from Kaggle).
2. Create a `data/` directory in the root of this project.
3. Extract and place the `mnist_train.csv` file inside the `data/` folder.

### 3. Build and Run
Compile and execute the engine using Cabal:
```bash
cabal build
cabal run

```

The execution will output the data loading progress, perform the Neural Network training loop over epochs (reporting accuracy), and finally run a side-by-side inference benchmark against the k-NN model.

```

Una vez que guardes el archivo, puedes registrar este cierre de proyecto en tu repositorio:

```bash
git add README.md
git commit -m "docs: update README with backpropagation, Cabal setup, and complete feature list"

```