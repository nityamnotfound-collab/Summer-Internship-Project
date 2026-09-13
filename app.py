import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("fruit_classifier.h5")
class_names = ['Pineapple', 'cherry', 'mango', 'plum', 'tomato', 'watermelon']  # update if you expanded classes

st.title("Fruit Classifier")
st.write("Upload a fruit image to identify it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((100, 100))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)

    st.write(f"**Prediction:** {predicted_class} ({confidence:.2%} confidence)")
