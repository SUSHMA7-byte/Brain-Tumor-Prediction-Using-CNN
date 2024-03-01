import streamlit as st
import keras
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = keras.models.load_model(r"C:\Users\welcome\Desktop\Projects\AIML_Projects\BrainTumorApp\brainTumor.h5", compile=False)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

def predict_single_image(image):
    x = np.array(image.resize((128, 128)))
    x = x.reshape(1, 128, 128, 3)
    res = model.predict_on_batch(x)
    classification = np.argmax(res)
    return res, classification

st.title("Brain Tumor Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if st.button("Make Prediction"):
    if uploaded_file:
        
        img = Image.open(uploaded_file)
        res, classification = predict_single_image(img)

        
        st.image(img, caption='Uploaded Image.', use_column_width=True)

        
        if classification == 1:
            st.write("Predicted class: Tumor not Detected")
        else:
            st.write("Predicted class: Tumor Detected")
        st.write(f"Confidence: {res[0][classification] * 100:.2f}%")
    else:
        st.warning("Please upload an image file.")
