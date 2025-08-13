# test.py (Final Corrected Version)

import gradio
import pickle
import numpy as np
from PIL import Image, ImageOps

# It seems you have a Keras model saved via pickle. Let's load it.
MODEL_FILENAME = 'model.pkl'  # <-- Make sure this is your file's name

try:
    with open(MODEL_FILENAME, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"FATAL ERROR: Model file not found at '{MODEL_FILENAME}'")
    exit()

def predict_digit(image):
    """
    This function takes an image, preprocesses it for a Keras CNN model,
    and returns the prediction.
    """
    # This initial check handles when the app first loads
    if image is None:
        return "Please draw a digit."

    # --- FIX #2: Handle the case where the user clicks the "Clear" button ---
    if image['composite'] is None:
        return "Please draw a digit."

    # Extract the numpy array from the dictionary
    pil_image = Image.fromarray(image['composite'])

    # Standard preprocessing
    gray_image = pil_image.convert('L')
    resized_image = gray_image.resize((28, 28))
    inverted_image = ImageOps.invert(resized_image)
    img_array = np.array(inverted_image) / 255.0

    # --- FIX #1: Reshape for a CNN model (1 sample, 28x28 size, 1 color channel) ---
    reshaped_image = img_array.reshape(1, 28, 28, 1)

    # Make prediction using the Keras model
    prediction_probabilities = model.predict(reshaped_image)
    
    # Create the dictionary of confidences for Gradio's Label output
    confidences = {str(i): float(prediction_probabilities[0][i]) for i in range(10)}

    return confidences

# Create and launch the Gradio interface
iface = gradio.Interface(
    fn=predict_digit,
    inputs="sketchpad",
    outputs=gradio.Label(num_top_classes=3),
    live=True,
    title="Digit Recognizer (CNN Model)",
    description="Draw a digit (0-9). The Keras CNN model will predict what it is."
)

iface.launch()