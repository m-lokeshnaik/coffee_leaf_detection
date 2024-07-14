import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf

# Assuming helper functions (clean_image, get_prediction, make_results) are in a separate file 'utils.py'
from utils import clean_image, get_prediction, make_results

# Function to load the model with error handling
@st.cache(allow_output_mutation=True)
def load_model(path):
    try:
        model = tf.keras.models.Sequential([
            tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        model.load_weights(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model(r"C:\Users\lokes\Desktop\coffee_leaf_detection\model_20.h5")

# Streamlit UI with error handling for uploaded file
st.title('Coffee Leaf Disease Detection')
st.write("Upload a coffee leaf image to get a prediction on its health.")

uploaded_file = st.file_uploader("Choose an image file (png or jpg)", type=["png", "jpg"])

if uploaded_file is not None:
    try:
        # Display informative messages during processing
        st.text("Preparing image...")
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(np.array(Image.fromarray(np.array(image)).resize((700, 400), Image.ANTIALIAS)), width=None)
        st.text("Making predictions...")

        # Preprocess image
        image = clean_image(image)

        # Get predictions and format results
        predictions, predictions_arr = get_prediction(model, image)
        result = make_results(predictions, predictions_arr)

        # Display clear and informative results
        class_labels = ["Healthy", "Disease 1", "Disease 2", "Disease 3"]  # Replace with actual class names
        st.success(f"The plant is predicted to be {class_labels[result['class_index']]} with {result['prediction']:.2f} confidence.")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Hide menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# import streamlit as st
# from PIL import Image
# import io
# import numpy as np
# import tensorflow as tf
# from utils import clean_image, get_prediction, make_results

# @st.cache(allow_output_mutation=True)
# def load_model(path):
#     model = tf.keras.models.load_model(path)
#     return model

# hide_streamlit_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# model = load_model(r"C:\Users\lokes\Desktop\coffee_leaf_detection\model_20.h5")

# st.title('coffee leaf Diesease Detection')
# st.write("Just Upload your coffee's Leaf Image and get predictions if the plant is healthy or not")

# uploaded_file = st.file_uploader("Choose a Image file", type=["png", "jpg"])

# if uploaded_file!= None:
#     progress = st.text("Crunching Image")
#     my_bar = st.progress(0)
#     i = 0

#     image = Image.open(io.BytesIO(uploaded_file.read()))
#     st.image(np.array(Image.fromarray(
#         np.array(image)).resize((700, 400), Image.ANTIALIAS)), width=None)
#     my_bar.progress(i + 40)

#     image = clean_image(image)

#     predictions, predictions_arr = get_prediction(model, image)
#     my_bar.progress(i + 30)

#     result = make_results(predictions, predictions_arr)

#     my_bar.progress(i + 30)
#     progress.empty()
#     i = 0
#     my_bar.empty()

#     st.write(f"The plant {result['status']} with {result['prediction']} prediction.")