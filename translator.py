
# coding: utf-8

# In[3]:


import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.layers import LSTM,Input
from keras.models import Model
import numpy as np
from keras.models import load_model
# @st.cache
def app():
    st.title("Translation APP !")

    # Upload the dataset and save as csv
    st.markdown("### Wrtie a sentence to translate.") 
    st.write("\n")
    
    value = st.text_input("Write a sentence in ENGLISH to translate")

    english_texts=[]
    french_texts=[]
    english_character=[]
    french_character=[]
    with open("fra.txt", "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in lines[:30000]:
        english_text,french_text,_=line.split("\t")
        english_texts.append(english_text)
        french_text = "\t" + french_text + "\n"
        french_texts.append(french_text)
    for i in english_texts:
        for c in i:
            if c not in english_character:
                english_character.append(c)
                english_character.sort()
    for j in french_texts:
        for c in j:
            if c not in french_character:
                french_character.append(c)
                french_character.sort()
    
    english_d={}
    for i in range(len(english_character)):
        english_d[english_character[i]]=i

    french_d={}
    for i in range(len(french_character)):
        french_d[french_character[i]]=i

    english_encoder_tokens = len(english_character)
    french_decoder_tokens = len(french_character)
    
    max_encoder_seq_length=0
    for i in english_texts:
        if len(i)>max_encoder_seq_length:
            max_encoder_seq_length=len(i)
    max_decoder_seq_length=0
    for i in french_texts:
        if len(i)>max_decoder_seq_length:
            max_decoder_seq_length=len(i)
            
    model = load_model("FrencTrans.h5")        
            
    latent_dim = 256         
    
    encoder_inputs = model.input[0]  # input_1
    encoder_outputs_1, state_h_enc_1, state_c_enc_1 = model.layers[2].output 
    encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output 
    encoder_states = [state_h_enc_1, state_c_enc_1,state_h_enc, state_c_enc]
    encoder_model_1 = keras.Model(encoder_inputs, encoder_states)
    decoder_inputs = model.input[1]  
    decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
    decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
    decoder_state_input_h1 = Input(shape=(latent_dim,),name="input_5")
    decoder_state_input_c1 = Input(shape=(latent_dim,),name="input_6")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c,decoder_state_input_h1,decoder_state_input_c1]
    decoder_lstm = model.layers[3]
    dec_o, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs[:2])
    decoder_lstm_1=model.layers[5]
    dec_o_1, state_h1, state_c1 = decoder_lstm_1(
    dec_o, initial_state=decoder_states_inputs[-2:])
    decoder_states = [state_h,state_c,state_h1,state_c1]
    decoder_dense = model.layers[6]
    decoder_outputs = decoder_dense(dec_o_1)
    decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)        
    
    reverse_input_char_index ={}
    for i in range(len(english_character)):
        reverse_input_char_index[i]=english_character[i]
    reverse_target_char_index ={}
    for i in range(len(french_character)):
        reverse_target_char_index[i]=french_character[i]

    def decode_sequence(input_seq):
        states_value=encoder_model_1.predict(input_seq)
        target_seq = np.zeros((1, 1, len(french_character)))
        target_seq[0, 0, french_d["\t"]] = 1.0
        flag=0
        sent=""
        while flag==0:
            output_tokens, h, c,h1,c1 = decoder_model.predict([target_seq] + states_value)
            sample = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sample]
            sent+=sampled_char
            if sampled_char == "\n" or len(sent) > max_decoder_seq_length:
                flag=1
            target_seq = np.zeros((1, 1, len(french_character)))
            target_seq[0, 0,sample] = 1.0
            states_value = [h, c,h1,c1]
        return sent
    
    reverse_input_char_index ={}
    for i in range(len(english_character)):
        reverse_input_char_index[i]=english_character[i]
    reverse_target_char_index ={}
    for i in range(len(french_character)):
        reverse_target_char_index[i]=french_character[i] 
    text=value
    k=len(text)
    m=0
    a=[]
    b=[]
    c=[]
    inpu=[]
    while m<k:
        for char in text[m]:
            for i in range(len(english_character)):
                if english_d[char]==i:
                    a.append(1)
                else:
                    a.append(0)
        for kp in a:
            b.append(kp)
        c.append(b)
        b=[]
        a=[]
        m=m+1
    while m<max_encoder_seq_length:
        for i in range(len(english_character)):
            if i==0:
                a.append(1)
            else:
                a.append(0)
        for kp in a:
            b.append(kp)
        c.append(b)
        b=[]
        a=[]
        m=m+1
    inpu.append(c)
    inpu=np.array(inpu)
    d=decode_sequence(inpu)
    st.subheader(d)
    

