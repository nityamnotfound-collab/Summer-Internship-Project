
import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Fruit Classifier")

@st.cache_resource
def load_assets():
    model = load_model("fruit_classifier.h5")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_assets()

st.title("Fruit Classifier")
st.write("Upload a fruit image to identify it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB").resize((100, 100))
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        pred = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(pred)]
        confidence = np.max(pred)

        st.success(f"**Prediction:** {predicted_class} ({confidence:.2%} confidence)")
    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
