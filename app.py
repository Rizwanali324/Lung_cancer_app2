import os 
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Function to load and preprocess the image
def preprocess_image(image):
    image = image.resize((100, 100))  # Resize to match model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to load and run inference on the model
def run_inference(image_array, model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Resize input tensor to accommodate the image
    interpreter.resize_tensor_input(input_details[0]['index'], [image_array.shape[0], 100, 100, 3])
    interpreter.allocate_tensors()
    
    interpreter.set_tensor(input_details[0]['index'], image_array.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# Define Streamlit app
def main():
    # Set page configuration with icon
    st.set_page_config(page_title="Lung Cancer Classification", page_icon="lung.png", layout='wide', initial_sidebar_state='expanded')
    st.sidebar.markdown("# Aibytec")
    
    st.sidebar.image('logo.jpg', width=200)
    st.title("Image Classification of Histopathological Images")

    # Custom CSS for animated background and different colored sections
    st.markdown("""
    <style>
    body {
        animation: gradientAnimation 15s ease infinite;
        background-size: 400% 400%;
        background-image: linear-gradient(45deg, #EE7752, #E73C7E, #23A6D5, #23D5AB);
    }

    @keyframes gradientAnimation {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }

    .dataset-info {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        color: #333333;
    }

    .instructions {
        background-color: rgba(255, 255, 255, 0.6);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        color: #333333;
    }

    .caution-note {
        background-color: rgba(255, 255, 255, 0.4);
        padding: 15px;
        margin-bottom: 15px;
        border-left: 6px solid #FF5733;
        border-radius: 5px;
        box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1);
        color: #333333;
    }

    .caution-note:last-child {
        margin-bottom: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Instructions for the user
    st.markdown("""
    ## Instructions
    
    1. Upload an image file (JPEG or PNG format) of a histopathological sample.
    2. Click the "Predict" button to get the Prediction result.
    """)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns([2, 1])  # Create two columns with different widths

    if uploaded_file is not None:
        # Preprocess image
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, caption="Uploaded Image", width=300)

        with col2:
            # Predict button
            if st.button("Predict"):
                # Preprocess the image
                image_array = preprocess_image(image)
                
                # Run inference
                output = run_inference(image_array, 'model.tflite')

                # Define class names
                class_names = [
                    "Colon benign tissue",
                    "Colon adenocarcinoma",
                    "Lung squamous cell carcinoma",
                    "Lung adenocarcinoma",
                    "Lung benign tissue"
                ]

                # Get predicted class index with the highest score
                predicted_class_index = np.argmax(output)
                
                # Display predicted class
                predicted_class = class_names[predicted_class_index]
                st.write("Predicted class:", predicted_class)
                
                # Display cautionary note as a popup message
                st.warning("""
                **Accuracy Disclaimer**: The predictions made by this app are based on a machine learning model and may not always be 100% accurate. Use the results as a supplementary tool and consult medical professionals for definitive diagnosis.
                """)

    
    # Dataset Information
    st.markdown("""
    ## Dataset Information
    
    This application uses a deep learning model to classify histopathological images of lung and colon tissues. The dataset used to train this model is publicly available on Kaggle:
    [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images).
    
    The dataset contains 5,000 images for each of the following classes:
    - Lung benign tissue
    - Lung adenocarcinoma
    - Lung squamous cell carcinoma
    - Colon adenocarcinoma
    - Colon benign tissue
    
    The images are histopathological samples stained with Hematoxylin and Eosin (H&E).
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
