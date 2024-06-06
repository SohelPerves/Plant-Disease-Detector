from flask import Flask, render_template, request
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

app = Flask(__name__)

# Define the model and class_names as global variables
potato_model = None

class_names = ["Early Blight", "Late Blight", "Healthy"]

def load_models():
    global potato_model
    potato_model_path = 'potato_model.h5'  
    
    potato_model = load_model(potato_model_path)

# Load the models before the first request
@app.before_request
def before_request():
    load_models()

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

# Load InceptionV3 model pre-trained on ImageNet
inception_model = InceptionV3(weights='imagenet')

def is_valid_leaf_image(image_path):
    # Load the image
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions using InceptionV3 model
    predictions = inception_model.predict(img_array)
    decoded_predictions = decode_predictions(predictions)

    # Print decoded predictions
    print(decoded_predictions)

    # Check if any of the top predictions indicate a plant or leaf
    for _, label, _ in decoded_predictions[0]:
        if 'plant' in label.lower() or 'leaf' in label.lower():
            return True

    return False


@app.route('/potato', methods=['GET', 'POST'])
def potato():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file part")
            return render_template('potato.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            print("No selected file")
            return render_template('potato.html', error='No selected file')

        if file:
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)

            # Ensure both models are loaded before making predictions
            load_models()

            # Check if the image is a valid plant leaf image
            if not is_valid_leaf_image(file_path):
                print("Invalid image Or a Healthy Leaf. Please upload a valid picture.")
                os.remove(file_path)  # Remove the invalid image
                return render_template('potato.html', error='Invalid image Or a Healthy Leaf. Please upload a valid picture.')

            print("Valid image. Continuing with predictions.")

            # Preprocess the image
            img_array = preprocess_image(file_path)

            # Make prediction using the potato model
            prediction = potato_model.predict(img_array)

            # Get the predicted class and confidence
            predicted_class_index = np.argmax(prediction)
            confidence = prediction[0][predicted_class_index]

            # Map class index to class name
            predicted_class_name = class_names[predicted_class_index]

            # Convert confidence to percentage
            confidence_percentage = confidence * 100
            suggestion = ""
            if predicted_class_name == "Early Blight":
                suggestion = "It looks like your potato plant has Early Blight. Consider applying fungicides and removing infected leaves."
            elif predicted_class_name == "Late Blight":
                suggestion = "Your potato plant may have Late Blight. Act promptly by applying fungicides and reducing humidity around the plants."

            # Display the result on the Potato page
            return render_template('potato.html', predicted_class=predicted_class_name, confidence=confidence_percentage, img_path=file_path, suggestion=suggestion)

    return render_template('potato.html')


@app.route('/suppliments')
def suppliments():
    # Add your code for the Suppliments page here
    return render_template('suppliments.html')

@app.route('/contact')
def contact():
    # Add your code for the Contact page here
    return render_template('contact.html')


@app.route('/predict', methods=['POST'])
def predict():
    # This route should handle predictions separately
    # You can use a similar approach as in the 'potato' route above
    # Ensure the model is loaded, preprocess the image, make predictions, and return the result
    return render_template('index.html')  # Placeholder for now

if __name__ == '__main__':
    app.run(debug=True)
