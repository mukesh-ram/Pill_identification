import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
model = load_model('pill_identifier_model.h5')

# Dictionary to map class indices to pill names
class_names = {
    0: 'aldigesic',
    1: 'calpol-650',
    2: 'paracip',
    3: 'zerodol-sp'
}

# Function to preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to make a prediction
def predict_pill(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_pill = class_names[predicted_class]

    # Display the image
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted Pill: {predicted_pill}')
    plt.show()

# Upload an image and make a prediction
uploaded_image_path = "C:\\Users\\mukes\\Downloads\\OIP.jpg"  # Replace with the path to your uploaded image
predict_pill(uploaded_image_path)
