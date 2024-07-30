import streamlit as st
from PIL import Image, ImageOps
import warnings
import tensorflow as tf
import os
import keras
import random
import numpy as np


warnings.filterwarnings("ignore")

model = keras.models.load_model("test.keras")

st.set_page_config(
    page_title = "CatsVDogs",
    page_icon = ":cat: :dog:",
    initial_sidebar_state = "auto"
)

st.write("Cats VS Dogs")

with st.sidebar:
    st.title("Cats Vs Dogs")
    st.image("cats-and-dogs.jpg")
    st.subheader("Dunno what pet you have,is it a cat or a dog, find out")

file = st.file_uploader("", type=["jpg", "png"])


def predict(img):
    img_height = 64
    img_width = 64

    class_names = ['cats', 'dogs']
    img =  tf.image.resize(img, [img_height, img_width], method = "bilinear")
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions_1 = model.predict(img_array)
    score_1 = tf.nn.softmax(predictions_1[0])

    return f"This image most likely belongs to {class_names[np.argmax(score_1)]} with a {100 * np.max(score_1):.2f} percent confidence."

if file is None:
    st.text("Upload a picture")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = predict(image)
    st.write(prediction)
