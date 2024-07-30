import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2


model = tf.keras.models.load_model("Classifier.keras")
cancer_names = ['adenocarcinoma left lower lobe',
 'large cell carcinoma left hilum',
 'normal',
 'squamous cell carcinoma left hilum']

st.write("Cancer Clasification")

file = st.file_uploader("", type=["jpg", "png"])

def pred(img):
    class_names = cancer_names
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    pred = model.predict(img_array)
    score = tf.nn.softmax(pred[0])

    return f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score):.2f} percent confidence."


if file is None:
    st.text("Upload a Picture")
else:
    image = Image.open(file)
    image = ImageOps.grayscale(image)
    st.image(image, use_column_width=True)
    prediction = pred(image)
    st.write(prediction)
