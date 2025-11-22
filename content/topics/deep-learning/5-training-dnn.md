---
category: "notes/deep-learning"
title: "Training deep neural networks in practice"
date: "2025-10-17"
description: ""
tags: []
---
## Weight Decay (L2 regularization in vanilla SGD)

We add a small penalty on large weights to the loss. This discourages over-reliance on any single weight, leading to smoother, simpler model.

## Dropout
Reduces overfitting.
During training, we randomly drop (set to 0) some neuron outputs with probability (1-p). This forces the network to not depend on any one path. Basically inject noise, training an ensemble of many subnetworks sharing parameters.

At inference time -> use all neurons, but multiply their outputs by p (to keep expected activation same as training).
Nowadays, large-scale models rely more on normalization/data augmentation, so dropout is only used for small datasets or RNNs / fully connected layers prone to overfitting.

## Parameter Initialization
Bad initialization -> gradients vanish (e.g. init all W to 0) or explode -> network can't learn.
Good initializations: 
- Xavier (Glorot) for tanh/sigmoid
- Kaiming (He) for ReLU

## Input Normalization & Batch Normalization
Before training, transform each input feature $x_i$ to reduce ill-conditioning (some features dominating others)

Batch normalization BN normalizes layer activations during training. 
It stabilizes training leading to faster convergence.

## Data Augmentation 
Artificially enlarge the dataset and inject useful noise (random crops for images, work dropout for text, pitch shift for audio, etc..)
It teaches model invariance and acts as regularization. 
Usually done online (at each epoch, with a different random transform).