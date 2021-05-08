import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import fnmatch
import tensorflow as tf
from time import sleep
import matplotlib.image as mpimg
import keras
import streamlit as st
import streamlit.components.v1 as stc

 
# loading the trained model
model = tf.keras.models.load_model("my_malaria_tf_model")
@st.cache()
def load_image(image_file):
    img = Image.open(image_file)
    return img 
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(inp_img):  
    class_names = ['Parasitized', 'Uninfected'] 
    inp_img= tf.expand_dims(inp_img, 0) # Create a batch
    predictions = model.predict(inp_img)
    score = tf.nn.softmax(predictions[0])
    accuracy= 100 * np.max(score)
    pred=class_names[np.argmax(score)]
    return pred,accuracy
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:blue;padding:15px"> 
    <h1 style ="color:white;text-align:center;">Prediction of Malaria</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 

    image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
    if image_file is not None:
        
            # To See Details
            # st.write(type(image_file))
            # st.write(dir(image_file))
        file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
        st.write(file_details)

        img = load_image(image_file)
        img_array = keras.preprocessing.image.img_to_array(img)

        im = cv2.resize(img_array, (128, 128), interpolation=cv2.INTER_CUBIC)
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"):
        p,a=prediction(im)
        print(p,a)        



if __name__=='__main__': 
    main()

