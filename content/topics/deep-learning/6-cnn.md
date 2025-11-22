---
category: "notes/deep-learning"
title: "Convolutional Neural Networks"
date: "2025-11-08"
description: ""
tags: []
---
## Convolution
It's a technique that allows us to detect patterns in a multi-dimensional data without flattening it into a 1D vector.

A **kernel** (or filter) is a small matrix of numbers (learnable weights) that detects certain patterns. It slides over our data, one small patch at a time. 

The **convolution** is the process of taking the dot product between the kernel and each region of the input. The process is a smaller grid called a **feature map** or **activation map**, containing the responses of the kernel at each location.