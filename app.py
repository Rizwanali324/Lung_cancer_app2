import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Define the directory containing subfolders with class names
base_dir = 'test_images'

# Function to preprocess the image
def preprocess_image(image):
    size = (256, 256)
    image = image.resize(size)
    image = image.convert('L')  # Convert to grayscale
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to make predictions
def predict(image):
    image_array = preprocess_image(image)
    predictions = model.predict(image_array)
    return predictions

# Function to load and display the first image from each class folder
def display_first_images(class_folders):
    num_folders = len(class_folders)
    fig, axes = plt.subplots(1, num_folders, figsize=(10, 3))

    for i, folder in enumerate(class_folders):
        class_name = os.path.basename(folder)
        images = os.listdir(folder)
        image_path = os.path.join(folder, images[0])  # Get the first image in the folder
        image = Image.open(image_path)
        image = image.resize((150, 150))  # Resize the image for display
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(class_name)
        axes[i].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

# Define Streamlit app
def main():
    global model  # Ensure model is global if you intend to access it inside functions
    
    # Load the pre-trained model
    model = tf.keras.models.load_model('model.h5')

    # Define the directory containing subfolders with class names
    base_dir = 'test_images'
    
    # Streamlit app setup
    st.set_page_config(page_title="Lung Cancer Classification", page_icon=":lungs:", layout='wide', initial_sidebar_state='expanded')
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

    # Get the list of subfolders (each representing a class)
    class_folders = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]

    display_first_images(class_folders)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image and prediction button in a column layout
        uploaded_image = Image.open(uploaded_file)
        col1, col2 = st.columns([2, 1])  # Adjust the width ratio as needed

        with col1:
            st.image(uploaded_image, caption='Uploaded MRI Image.', width=350)
        
        with col2:
            # Prediction button
            if st.button('Predict'):
                # Get predictions
                predictions = predict(uploaded_image)
                predicted_class_idx = np.argmax(predictions)

                # Define class names (ensure it matches your model's output)
                class_names = [
                    "Colon benign tissue",
                    "Colon adenocarcinoma",
                    "Lung squamous cell carcinoma",
                    "Lung adenocarcinoma",
                    "Lung benign tissue"
                ]

                # Check if predicted_class_idx is within bounds
                if 0 <= predicted_class_idx < len(class_names):
                    predicted_class = class_names[predicted_class_idx]
                    st.write(f"Prediction: {predicted_class}")

                    # Display caution warning
                    st.warning("""
                    Accuracy Disclaimer: The predictions provided by this app are generated by a machine learning model currently in the research phase and may not always be fully accurate. Please use these results as supplementary information only. Always consult with medical professionals for a definitive diagnosis and doctor/professional advice.
                    """)

                else:
                    st.write("Invalid prediction index.")

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
