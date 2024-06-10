import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

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
    st.title("Image Classification  histopathological images of lung and colon tissues")

    # Information about the dataset
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
    """)

    # Instructions for the user
    st.markdown("""
    ## Instructions
    1. Upload an image file (JPEG or PNG format) of a histopathological sample.
    2. Click the "Predict" button to get the classification result.
    """)

    # File uploader
    st.sidebar.title("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess image
        image = Image.open(uploaded_file)
        image = image.resize((200, 200))  # Resize the image to 200x200 pixels
        st.image(image, caption="Uploaded Image", width=500)
        image_array = preprocess_image(image)

        # Run inference
        output = run_inference(image_array, r'C:\Users\Moazzam\Desktop\Rizwan\freelancing\Lung_cancer_app2\model.tflite')

        # Define class names
        class_names = [
            "Lung benign tissue",
            "Lung adenocarcinoma",
            "Lung squamous cell carcinoma",
            "Colon adenocarcinoma",
            "Colon benign tissue"
        ]

        # Get predicted class index with the highest score
        predicted_class_index = np.argmax(output)
        predicted_score = output[0][predicted_class_index]
        
        # Display only the predicted class and score with the highest score
        predicted_class = class_names[predicted_class_index]
        st.write("Predicted class:", predicted_class)
        st.write("Predicted Score:", predicted_score)

if __name__ == "__main__":
    main()


