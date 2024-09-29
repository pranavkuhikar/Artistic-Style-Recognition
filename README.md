# Artistic Style Recognition Using Convolutional Neural Networks (CNNs)


# Project Overview
This repository contains the implementation of my MSc dissertation titled "A Convolutional Neural Network Approach to Artistic Style Recognition". The project explores the application of deep learning and Convolutional Neural Networks (CNNs) to classify artistic styles, offering a scalable solution for art historians, educators, and enthusiasts.

Art analysis traditionally relies on subjective evaluations from experts, making it difficult to scale for large collections. By leveraging CNNs, this project seeks to automate the recognition of various artistic styles with a focus on improving accuracy through advanced deep learning techniques.

# Key Features
End-to-End Deep Learning Pipeline: The project uses CNNs to automatically classify artistic styles.
Advanced Techniques: Includes methods such as transfer learning, pyramid fusion, and multi-branch networks to enhance accuracy and model performance.
Patch-Based and Sub-Region Analysis: For improved style recognition by focusing on smaller image segments.
Accuracy: The project achieved an accuracy of 64.5% with CNN models optimized through hyperparameter tuning.

# Model Development
The project implemented and tested multiple CNN models:

Baseline Models: Including ResNet, Xception, DenseNet, MobileNet, and Inception models, pretrained on ImageNet.
Custom Models: Variants were developed using optimizers like Adam and Stochastic Gradient Descent (SGD), and additional techniques like batch normalization and dropout layers to enhance generalization and reduce overfitting.
Dataset
The dataset used is from Wiki-Art, comprising 75,960 images across 19 well-represented classes. Extensive preprocessing, including augmentation techniques (resizing, normalization, etc.), was performed to improve model performance.

# Results
Best Model: The best-performing model was based on the Xception architecture, achieving an accuracy of 64.5% in classifying different artistic styles.
Challenges: Difficulties arose in distinguishing closely related styles and handling imbalanced data, which were mitigated through techniques such as patch analysis and pyramid fusion.

# Tools and Technologies
TensorFlow and Keras for building and training CNN models.
Python for data processing and experimentation.
VS Code and Jupyter Notebooks for development and documentation.

# Requirements

# A) Software Requirements

1. Download and install the Nvidia graphics driver compatible with the system’s GPU to
ensure TensorFlow can leverage GPU acceleration. In this case, CUDA 11.8.0.
2. Verify the compatibility of all software components with TensorFlow by consulting the
official TensorFlow website. Install tensorflow version 2.10.
3. Install Visual Studio Community 2019 to obtain the necessary Visual C++ redistributables
required by TensorFlow on Windows.
4. Install the CUDA Toolkit 11.8.0, which provides the GPU acceleration capabilities needed
by TensorFlow.
5. Install cuDNN 8.6 to enhance neural network performance via GPU acceleration.
6. Install Anaconda3 to manage project environments and simplify package management
effectively.
7. Create a virtual environment to isolate and manage the TensorFlow project’s dependencies.
8. Install TensorFlow within the virtual environment to ensure it utilizes the correct settings
and dependencies.
9. Verify the TensorFlow installation by testing its GPU recognition to confirm proper setup
and configuration.

# B) Hardware Requirements
1. The project utilized the following hardware configuration: an NVIDIA GeForce RTX
3060 GPU, driven by driver version 555.99 and CUDA version 12.5, supported by a
compilation environment facilitated through CUDA toolkit version 11.8. This setup
provided the computational power required for the deep learning tasks outlined in the
research.
2. Device used: Lenovo Legion 5 Pro 16ACH6H 2021
3. GPU configuration: NVIDIA GeForce RTX 3060, 6 GB
4. Processor: AMD Ryzen 7 5800H, 8C/16T
5. Memory: 16 GB DDR4-3200 (2x 8GB DIMMs) – 2xDIMMs


# Future Work
Improve model accuracy beyond 70% by incorporating Generative Adversarial Networks (GANs).
Extend the model to support real-time artwork analysis in museum applications.
Integrating it with Chatgpt API to enrich user experience.

# Acknowledgements
Special thanks to Prof. Peter Hall and Prof. Nello Cristianini at the University of Bath for their guidance and support throughout this project.
