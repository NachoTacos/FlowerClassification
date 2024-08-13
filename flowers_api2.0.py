import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 
import pandas as pd
from io import StringIO

st.header('Image Classification Model')
model = load_model('flowermodel.keras')
data_cat = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
img_height = 180
img_width = 180

uploaded_file = st.file_uploader("Choose an image...", type =["jpg","jpeg","png"])

if uploaded_file is not None:
    image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height,img_width))
    img_arr = tf.keras.utils.array_to_img(image_load)
    img_bat=tf.expand_dims(img_arr,0)
    
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
    
    st.image(uploaded_file, width=200)
    st.write('Flowerin image is ' + data_cat[np.argmax(score)])
    st.write('With accuracy of ' + str(np.max(score)*100))
    
                                 
                



