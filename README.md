# MedMNIST Breast Image Classification

Welcome to my final year project repository!

In this project, I utilised the MedMNIST Breast Cancer dataset for binary image classification using Convolutional Neural Networks (CNN). The goal of the project was to develop a model that can accurately classify images as either benign or malignant based on breast tissue. Below, you can find more details on the dataset, the model architecture, and how I created the user interface with Streamlit.

Project Overview

I used the MedMNIST breast dataset which contains preprocessed medical images for various classification tasks. I focused on the breast cancer category for this project. The dataset contains labeled breast cancer images in a binary format and they are categorized in two classes as 0 and 1. 0 for malignant images and 1 for benign and normal images.

The CNN model achieved an accuracy of 82% on the test dataset.

Task: Binary-Class classification (Benign or Malignant).

Testing
The model was successfully tested on new and unseen data using the Streamlit user interface which I created allowing users to upload images and get real-time predictions. However, the accuracy is not 100%, so the model can make mistake predictions. 

Dataset Details
Source: The BreastMNIST dataset is sourced from the work of Walid Al-Dhabyani, Mohammed Gomaa, et al., in the paper Dataset of breast ultrasound images.

Data Modality: Breast ultrasound images.

Number of Samples
Total Samples: 780 images

Training Samples: 546 images

Validation Samples: 78 images

Test Samples: 156 images

License: CC BY 4.0 (Creative Commons Attribution 4.0 International License).

Model Architecture

I built a Convolutional Neural Network (CNN) for the classification task. The architecture of the model consists of:

Convolutional Layers for feature extraction from the images.
MaxPooling layers to reduce the dimensionality of the data and prevent overfitting.
Fully Connected Layers to make the final binary predictions (Benign or Malignant).
Activation Functions which is ReLU (Rectified Linear Unit) for the hidden layers and Sigmoid for the binary output layer.

The model was trained using TensorFlow/Keras and I used the Adam optimizer for better convergence.

Resources Used

Libraries
TensorFlow/Keras for model building and training.
NumPy and Pandas for data manipulation and preprocessing.
Streamlit for the interactive user interface.
Matplotlib for visualizations.

Development Environment
  Python 3.x
  Google Colab 
  Streamlit for deploying the model as a web app.

Streamlit User Interface
To make the model easily accessible to users, I created a Streamlit based web application.

The interface allows users to:
1. Upload new, unseen breast cancer images.
2. Get predictions for whether the image is benign or malignant.
3. View the model's confidence level for each prediction.

[Here is the link to my Streamlit page](streamlit-link-here) where you can test the model by uploading your own image.

How to Run the Project Locally
To run the project locally you will first need to clone this repository and install the required dependencies.

Download the app.py:
```

```

Install the required dependencies:

```
pip install -r requirements.txt
```

Download the model and save it in the same directory to the app.py file
```
https://github.com/18430349/medmnist-image-classification/blob/main/breast_cancer_cnn_model.keras
```
3. Run the Streamlit app:

```
streamlit run app.py
```

4. Access the app in your browser at `http://localhost:8501`.

## Conclusion

This project demonstrates the potential of machine learning for medical image classification tasks, particularly in diagnosing breast cancer. The model provides a reliable, automated way to predict whether breast tissue is benign or malignant, which can be used as an aid for medical professionals.

Feel free to explore the code, and don't hesitate to raise any issues or contribute!

---

Let me know if you'd like any more details or adjustments to the text!
