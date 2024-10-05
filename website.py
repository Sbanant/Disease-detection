import numpy as np
from flask import Flask, request, render_template
import pickle
from keras.models import load_model
import cv2

app = Flask(__name__)

# Load the trained model (Multinomial Naive Bayes)
# Update the path to the correct model directory
model = pickle.load(open('path_to_your_model_directory/model.pkl', 'rb'))

# Ensure the vectorizer is loaded from the same file used during training
# Update the path to the correct vectorizer directory
vectorizer = pickle.load(open('path_to_your_vectorizer_directory/vectorizer.pkl', 'rb'))

# Load the image classification model
# Update the path to the correct directory where the model.h5 is stored
model2 = load_model("path_to_your_model_directory/model.h5", compile=False)
class_names = open("path_to_your_model_directory/labels.txt", "r").readlines()

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/skin")
def skin():
    return render_template('skin.html')

@app.route("/predict")
def prediction():
    return render_template('choose.html')

@app.route('/prediction', methods=['POST'])
def predict():
    try:
        input_features = [x for x in request.form.values()]  

        input_vector = vectorizer.transform(input_features)

        print("Input Vector Shape:", input_vector.shape)

        if input_vector.shape[1] != 1549:
            return render_template('choose.html', prediction_text='Error: Invalid number of features')

        prediction = model.predict(input_vector)

        output = prediction[0]

        return render_template('choose.html', prediction_text='Disease Diagnosis: {}'.format(output))
    except Exception as e:
        return render_template('choose.html', prediction_text='Error: {}'.format(str(e)))

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' in request.files:
            image = request.files['image']

            # Check if image is empty
            if not image:
                return "Uploaded image is empty"

            image_data = image.read()

            # Check if image_data is empty or None
            if not image_data:
                return "Image data is empty"

            img_array = np.frombuffer(image_data, np.uint8)

            # Check if img_array is empty
            if img_array.size == 0:
                return "Image data is empty"

            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Check if img is empty
            if img is None:
                return "Failed to decode the image"

            # Check if image dimensions are valid
            if img.shape[0] == 0 or img.shape[1] == 0:
                return "Invalid image dimensions"

            # Resize the image
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
            img = img / 255  # Normalize the image to [0, 1]

            # Make a prediction using the image classification model (model2)
            prediction = model2.predict(img)
            index = np.argmax(prediction)
            class_name = class_names[index]
            class_name = class_name[2:]

            return render_template('skin.html', prediction_text='Disease Diagnosis: {}'.format(class_name))
    except Exception as e:
        return "Error: {}".format(str(e))

    return "No image uploaded"

if __name__ == "__main__":
    app.run()
