
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('breast_cancer_cnn_model.keras')
# Set page configuration with default styling
st.set_page_config(page_title="Breast Cancer Tumor Classifier", layout="wide")

# Sidebar with a welcome message and upload instructions
st.sidebar.title("Welcome")
st.sidebar.write("Please upload your ultrasound image here.")

# Main title and instructions
st.title("Breast Cancer Tumor Classifier")
st.write("Upload an ultrasound image of a breast tumor. The input image will be checked for validity and appropriate resolution.")

# File uploader
uploaded_image = st.file_uploader("Upload Ultrasound Image", type=["jpg", "jpeg", "png"])

def is_valid_image(uploaded_file):
    
# To check if the uploaded file is a valid image.
    
    try:
        # Attempt to open and verify the image.
        img = Image.open(uploaded_file)
        img.verify()
        return True
    except Exception:
        return False

if uploaded_image is not None:
    if not is_valid_image(uploaded_image):
        st.error("The uploaded file is not a valid image. Please upload a valid ultrasound image.")
    else:
        try:
            # Open the image for further processing
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Preprocessing the image...")
            
            # Convert image to grayscale
            image = image.convert('L')
            image_array = np.array(image)
            
            # Check that image dimensions are reasonable
            if image_array.shape[0] < 28 or image_array.shape[1] < 28:
                st.error("Image dimensions are too small. Please upload an ultrasound image with sufficient resolution.")
            else:
                # Resize image to the required 28x28
                processed_image = cv2.resize(image_array, (28, 28))
                
                # Prepare the image for the neural network model: add channel dimension and normalize
                processed_image = processed_image.reshape(1, 28, 28, 1).astype('float32') / 255.0
                
                # Load the pretrained neural network model
                model = load_model("breast_cancer_cnn_model.keras")
                
                with st.spinner('Processing...'):
                    prediction = model.predict(processed_image)
                
                confidence = prediction[0][0]
                result = "Benign" if confidence > 0.5 else "Malignant"
                st.write(f"Prediction: **{result}**")
                st.write(f"Confidence: **{(confidence if result == 'Benign' else 1 - confidence) * 100:.2f}%**")
                st.success('Done!')
        except Exception as e:
            st.error(f"Error processing the image: {e}")
else:
    st.info("Please upload an ultrasound image to get started.")
