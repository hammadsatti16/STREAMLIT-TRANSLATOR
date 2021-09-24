import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports 
from multipage import MultiPage
import translator, About, Record

# Create an instance of the app 
app = MultiPage()

# Title of the main page
display = Image.open('Logo.png')
display = np.array(display)

col1, col2 = st.columns(2)
col1.image(display, width = 600)
# Add all your application here
app.add_page("Text Translation", translator.app)
app.add_page("Voice Translation", Record.app)
app.add_page("About", About.app)

# The main app
app.run()
