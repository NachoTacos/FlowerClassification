import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 

st.header('Image Classification Model')
model = load_model('flowermodel.keras')
data_cat = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
img_height = 180
img_width = 180
image =st.text_input('Enter Image name','PosibleRosa.jpg')

image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write('Flowerin image is ' + data_cat[np.argmax(score)])
st.write('With accuracy of ' + str(np.max(score)*100))