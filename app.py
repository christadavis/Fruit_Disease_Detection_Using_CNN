import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model('C:/Users/user/OneDrive/Desktop/Deep_Learning_Project/Fruit Disease Detection Dataset.v4-presentation.tensorflow/model/fruit_disease_detector.h5')

# Define class indices manually (match this with your training dataset classes)
class_indices = {0: 'It is Non-Diseased Fruit', 1: 'It is Diseased Fruit'}  # Update this dictionary with actual class labels

# Set the title of the app
st.title('Fruit Disease Detection')

# Display a brief description
st.write("Upload an image of the fruit, and the model will predict if the fruit is diseased or healthy.")

# Image upload section
uploaded_file = st.file_uploader("Choose a fruit image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    # Preprocess the image (resize and normalize as done during training)
    img = np.array(image)
    
    # Check if the image is 3-channel; convert if not
    if len(img.shape) == 2 or img.shape[2] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (128, 128))  # Resize to the size used for training
    img = img / 255.0  # Normalize to the range [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension (required for the model)

    # Make prediction
    with st.spinner('Analyzing the image...'):
        prediction = model.predict(img)
    
    predicted_class = np.argmax(prediction)  # Get the index of the highest probability
    predicted_label = class_indices.get(predicted_class, "Unknown")  # Map index to label

    # Display the result
    st.success(f"The model predicts: {predicted_label}")
   # st.write(f"Predicted Class Index: {predicted_class}")
    #st.write(f"Class Indices: {class_indices}")
