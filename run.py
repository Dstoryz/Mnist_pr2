
# import pickle
import streamlit as st
from streamlit_drawable_canvas import st_canvas
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

import torch

from torch.nn.functional import interpolate

import scipy.misc


canvas_result = st_canvas(
        fill_color="rgba(165, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=5,
        stroke_color="#000000",
        background_color="#FFFFFF",
        background_image=None,
        update_streamlit=True,
        height=28*5,
        width=28*5,
        drawing_mode="freedraw",
        point_display_radius= 0,
        key="canvas"
)
    

def canvas_to_tensor(canvas):
    """
    Convert Image of RGBA to single channel B/W and convert from numpy array
    to a PyTorch Tensor of [1,1,28,28]
    """
    img = canvas.image_data
    img = img[:, :, :-1] # Ignore alpha channel
    # print(img.shape)
    img = img.mean(axis=2) # Convert to &&W
    img = (255-img)/255 # Range [0,1]
    print(img.shape)
    # img = img*2 - 1. # Range [-1,1]
    img = torch.FloatTensor(img)
    tens = img.unsqueeze(0).unsqueeze(0) # Add Batch and Channel Dimension
    tens = interpolate(tens, (28, 28)) # Image Resizing
    print(1, tens.shape)
    return tens


def predict_digit(sample):
    path_model = Path("saved_model", "1")
    model = tf.keras.models.load_model(path_model)
    prediction = model(sample[None, ...])[0]
    ans = np.argmax(prediction)


    print('Predicted number: {}'.format(ans))
    return ans

        

       
button1 = st.button("Проверка")
if button1:
    if canvas_result.image_data is not None:
        z = canvas_result
        z1 = canvas_to_tensor(z)
        z2 = z1.reshape((-1, 28, 28, 1))

        home = np.asarray(z2)
        
        st.write('Предсказанная цифра: ', format(predict_digit(home[0, ...])))
        # st.write(predict_digit(home[0, ...]))
        # st.write(
        # f'<Предсказанная цифра: "{predict_digit(home[0, ...])}">',
        # unsafe_allow_html=True,
        # )
       
        
        