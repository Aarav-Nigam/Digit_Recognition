import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

def predictDigit(image):
    model = tf.keras.models.load_model("mnist_model.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred= model.predict(img)
    result = np.argmax(pred[0])
    return result

# Streamlit 
st.set_page_config(page_title='Handwritten Digit Recognition', layout='centered')

st.title('Handwritten Digit Recognition(MNIST Based)')

drawing_mode = "freedraw"
stroke_width = st.slider('Select Stroke Width', 1, 30, 15)
stroke_color = '#FFFFFF' 
bg_color = '#000000'

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# Add "Predict Now" button
if st.button('Predict'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        input_image.save('temp/temp.png')
        img = Image.open("temp/temp.png")
        res = predictDigit(img)
        st.header('Prediction: ' + str(res))
    else:
        st.header('Please draw a digit on the canvas.')

# Define the footer HTML content with CSS for sticky positioning
footer = '''
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
}
</style>
<div class="footer">
    <p>Developed  by <a style='display: block; text-align: center; color:black ;font-weight: bold; font-size:20px' href="https://portfolio-aarav.netlify.app/" target="_blank">~Aarav Nigam</a></p>
</div>
'''

# Display the footer using st.markdown
st.markdown(footer, unsafe_allow_html=True)