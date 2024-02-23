# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Function to load the saved model
def load_model():
    model = tf.keras.models.load_model('covid_detection_model.h5')  # Update with your actual path
    return model

# Function to preprocess an image for prediction
def preprocess_image(image):
    # Resize the image to the desired size (224, 224)
    image = image.resize((224, 224))

    # Convert the PIL image to a NumPy array
    img_array = tf.keras.preprocessing.image.img_to_array(image)

    # Expand the dimensions to match the model input shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the pixel values to the range [0, 1]
    img_array /= 255.0

    return img_array

# Function to make predictions using the loaded model
# Function to make predictions using the loaded model
def predict(image, model):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)[0]  # Assuming a single prediction
    predicted_class = np.argmax(predictions)
    c=['Covid', 'Normal', 'Viral Pneumonia']
    return c[predicted_class]

# Streamlit app
def main():
    st.title("Covid-19 detection Using Deep Learning")

    uploaded_file = st.file_uploader("Choose an image of chest x-ray", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Load the model
        model = load_model()

        # Make predictions when the user clicks the button
        if st.button("Predict"):
            # Perform prediction
            predicted_class = predict(image, model)
            st.write("Predicted Class:")
            st.write(predicted_class)

if __name__ == '__main__':
    main()
