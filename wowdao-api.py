import requests
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input


app = Flask(__name__)
CORS(app)
# api = Api(app)


@app.route('/predict', methods=['POST'])
def predict_label():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"})

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected image"})

    # Load the trained model
    model = load_model('covid_detection_model.h5') 

    # Load and preprocess the new image
    # image_path = '/content/Data/test/COVID19/COVID19(460).jpg'
    # input_size = (128, 128)

    # image = load_img(image_path, target_size=input_size)

    image = Image.open(file)
    input_size = (128, 128)
    image = image.resize(input_size)

    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)

    # Expand dimensions to match the input shape expected by the model
    image_array = np.expand_dims(image_array, axis=0)

    # Make predictions
    predictions = model.predict(image_array)

    # Assuming predictions is a one-hot encoded array, you can get the predicted class index
    predicted_class_index = np.argmax(predictions)

    class_labels = ["COVID19", "NORMAL", "PNEUMONIA"]
    predicted_class_label = class_labels[predicted_class_index]

    print("Predicted Class:", predicted_class_label)
    print("Predictions:", predictions)

    return jsonify({"predicted_class": predicted_class_label})


   

# api.add_resource(TestClass, "/next")    

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')