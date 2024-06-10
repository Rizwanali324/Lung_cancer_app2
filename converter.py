import tensorflow as tf
import os

# Load the Keras model
model_path = 'Lung_cancer_app2/lungs_cancer000.h5'
keras_model = tf.keras.models.load_model(model_path)

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

# Apply post-training quantization to reduce model size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Target size in bytes (25 MB)
target_size_bytes = 25 * 1024 * 1024

# Save the TensorFlow Lite model
tflite_model_path = 'Lung_cancer_app2/model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# Check the size of the converted model
actual_size_bytes = os.path.getsize(tflite_model_path)

# If the actual size exceeds the target size, try additional optimization
if actual_size_bytes > target_size_bytes:
    print("Actual model size exceeds the target size. Applying additional optimization...")

    # Additional optimization steps here

    # Write the optimized model to the same file path
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

print("TensorFlow Lite model saved at:", tflite_model_path)
