
# coding: utf-8

# In[2]:


import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
# @st.cache
def app():
    st.markdown("# About")
    model = load_model("C:/Users/hamma/Downloads/Medina/pages/Translator.h5")   
    x=model.summary()
    st.markdown("#### This webiste is translates the english sentence to french sentence the RNN based LSTM encoding decoding model is used.")
    
    

