import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image


model_NN = load_model(r'C:\Users\mwael\OneDrive\Desktop\home\course\pneumonia\pneunonia_expert.keras')

st.title("Pneumonia Detection App🫁")

uploaded_file = st.file_uploader("Please upload a chest X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)


    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((128, 128))  
    ImgArray = image.img_to_array(img)
    ImgArray = np.expand_dims(ImgArray, axis=0)
    prediction = model_NN.predict(ImgArray)
    
    if prediction < 0.5:

        st.subheader("The model predicts: **Normal**  ALHAMDULLAH ")

    else:

        st.subheader("The model predicts: **Pneumonia**")