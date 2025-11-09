E-Waste Classification using CNN
Project Overview

This project focuses on classifying different types of electronic waste (e-waste) using Convolutional Neural Networks (CNN). The main aim is to promote smart and sustainable recycling practices by automatically identifying e-waste materials such as plastic, metal, glass, and circuit boards through image classification. The model helps in sorting and recycling e-waste in an efficient and eco-friendly manner.

Problem Statement

Electronic waste is increasing rapidly due to continuous technological advancement and improper disposal of electronic products. This leads to pollution, loss of reusable materials, and harm to the environment. There is a need for an automated system that can identify and classify e-waste materials accurately.
This project attempts to solve that problem using a CNN-based image classification approach.

Objective

To build a machine learning model that can classify e-waste images into different categories.

To improve waste sorting and recycling through artificial intelligence.

To support environmental sustainability and reduce the impact of electronic waste.

Dataset

The dataset used in this project is the E-Waste Image Dataset, which contains images of various e-waste materials categorized into multiple classes such as plastic, metal, glass, and circuit boards. The dataset is suitable for training computer vision models to identify e-waste components.

Tools and Technologies Used

Python

TensorFlow and Keras

NumPy and Matplotlib

Google Colab

GitHub for version control

Methodology

The dataset was collected and organized into training, validation, and testing folders.

Images were preprocessed using normalization and resizing.

A Convolutional Neural Network (CNN) model was built using TensorFlow/Keras.

The model was trained for 10 epochs and evaluated on the test dataset.

The performance of the model was measured using accuracy metrics.

Results

Test Accuracy achieved: 52%

Optimizer used: Adam

Loss function: Categorical Crossentropy

Epochs: 10
The model successfully classified different types of e-waste images, showing good potential for improving with more data and fine-tuning.

Future Scope

Increase accuracy using more image data and data augmentation techniques.

Implement object detection to identify multiple waste types in one image.

Deploy the model as a web or mobile application for real-time use.

Integrate with IoT-based recycling systems.

Files in the Project

e_waste_classification.ipynb – Main Jupyter Notebook file containing the full code.

ewaste_cnn_model_improved_10epochs.h5 – Saved trained CNN model.

archive(10).zip – Dataset used for training and testing.

Conclusion

This project demonstrates how artificial intelligence and computer vision can contribute to solving real-world environmental problems. The developed CNN model can help improve e-waste management and promote sustainable recycling practices.
