# # from flask import Flask, request, jsonify
# # import tensorflow as tf
# # from PIL import Image
# # import numpy as np
# # import requests
# # from io import BytesIO
# # import logging
# # import sys

# # app = Flask(__name__)

# # # Configure logging
# # logging.basicConfig(level=logging.DEBUG)
# # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# # # Load model
# # model = tf.keras.models.load_model('../covid19_model.h5')

# # # Load class names
# # with open('C:\lung-disease-detection-master\lung-disease-detection-master\classes.txt', 'r', encoding='utf-8') as f:
# #     class_names = f.read().splitlines()

# # def preprocess_image(image):
# #     try:
# #         image = image.resize((64, 64))  # Resize to match your model's input shape
# #         image = np.array(image)
# #         image = np.expand_dims(image, axis=0)  # Add batch dimension
# #         image = image / 255.0  # Normalize if necessary
# #         return image
# #     except Exception as e:
# #         app.logger.error(f"Error during image preprocessing: {e}")
# #         raise


# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     data = request.get_json()
# #     if 'url' not in data:
# #         return jsonify({'error': 'No URL provided'}), 400

# #     image_url = data['url']

# #     try:
# #         response = requests.get(image_url)
# #         response.raise_for_status()  # Ensure we notice bad responses
# #         image = Image.open(BytesIO(response.content))
# #     except requests.RequestException as e:
# #         app.logger.error(f"Error fetching image: {e}")
# #         return jsonify({'error': 'Error fetching image'}), 400
# #     except Exception as e:
# #         app.logger.error(f"Error processing image: {e}")
# #         return jsonify({'error': 'Error processing image'}), 400

# #     try:
# #         app.logger.info("Preprocessing image...")
# #         image = preprocess_image(image)
# #         app.logger.info(f"Image preprocessed successfully. Shape: {image.shape}, Type: {image.dtype}")

# #         app.logger.info("Making prediction...")
# #         predictions = model.predict(image, verbose=0)
# #         app.logger.info(f"Prediction made successfully. Predictions: {predictions}")

# #         predicted_class_index = np.argmax(predictions)
# #         app.logger.info(f"Predicted class index: {predicted_class_index}")

# #         predicted_class_name = class_names[predicted_class_index]
# #         app.logger.info(f"Predicted class name: {predicted_class_name}")

# #         probability = float(np.max(predictions))
# #         app.logger.info(f"Probability: {probability}")

# #         second_predicted_class_index = np.argsort(predictions[0])[-2]
# #         app.logger.info(f"Second predicted class index: {second_predicted_class_index}")

# #         second_predicted_class_name = class_names[second_predicted_class_index]
# #         app.logger.info(f"Second predicted class name: {second_predicted_class_name}")

# #         second_probability = float(predictions[0][second_predicted_class_index])
# #         app.logger.info(f"Second probability: {second_probability}")

# #     except Exception as e:
# #         app.logger.error(f"Error during prediction: {str(e).encode('utf-8', 'ignore').decode('utf-8')}")
# #         return jsonify({'error loc 2': 'Error during prediction'}), 500

# #     return jsonify([
# #         {
# #             'class': predicted_class_name,
# #             'probability': probability
# #         },
# #         {
# #             'class': second_predicted_class_name,
# #             'probability': second_probability
# #         }
# #     ])

# # if __name__ == '__main__':
# #     app.run(debug=True)


# from flask import Flask, request, jsonify, send_from_directory
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import requests
# from io import BytesIO
# import logging
# import sys

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# # Load model
# model = tf.keras.models.load_model('../covid19_model.h5')

# # Load class names
# with open(r'C:\\lung-disease-detection-master\\lung-disease-detection-master\\classes.txt', 'r', encoding='utf-8') as f:
#     class_names = f.read().splitlines()

# def preprocess_image(image):
#     try:
#         image = image.resize((64, 64))  # Resize to match your model's input shape
#         image = np.array(image)
#         image = np.expand_dims(image, axis=0)  # Add batch dimension
#         image = image / 255.0  # Normalize if necessary
#         return image
#     except Exception as e:
#         app.logger.error(f"Error during image preprocessing: {e}")
#         raise

# @app.route('/')
# def home():
#     return "Welcome to the Lung Disease Detection API."
# @app.route('/frontend')
# def frontend():
#     return send_from_directory('.', 'static/index.html')
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     # Your prediction logic here
# #     input_data="https://img.medscapestatic.com/pi/meds/ckb/94/38694tn.jpg"
# #     prediction = model.predict(input_data)  # Example: Replace with actual prediction logic
# #     print(f"Prediction: {prediction}")  # Add this line to display prediction in terminal
# #     return jsonify({'prediction': prediction})
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     data = request.get_json()
# #     print(f"Received data: {data}")  # To check what data you're receiving in the terminal

# #     url = data.get('url')
# #     if not url:
# #         return jsonify({"error": "URL is missing"}), 400

# #     # Simulated prediction (you can replace this with your actual model prediction logic)
# #     result = {"prediction": "No disease detected"}
    
# #     print(f"Prediction result: {result}")  # Output the prediction to console

# #     return jsonify(result), 200
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     url = data.get('url')
#     if not url:
#         return jsonify({"error": "URL is missing"}), 400

#     # Log that the request has been received
#     print(f"Received URL: {url}")
    
#     # Simulate model processing or add actual model code here
#     # Example:
#     # model_result = model.predict(url)
#     # For now, we'll simulate with a dummy response
    
#     result = {"prediction": "No disease detected"}
    
#     print("Prediction made.")
    
#     return jsonify(result), 200
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     data = request.get_json()
# #     url = data.get('url')
    
# #     # Add logic for prediction here
# #     # Example dummy response
# #     if not url:
# #         return jsonify({"error": "URL is missing"}), 400
    
# #     # Simulate prediction result
# #     result = {"prediction": "No disease detected"}
    
# #     return jsonify(result), 200

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     data = request.get_json()
# #     if 'url' not in data:
# #         return jsonify({'error': 'No URL provided'}), 400

# #     image_url = data['url']

# #     try:
# #         response = requests.get(image_url)
# #         response.raise_for_status()  # Ensure we notice bad responses

# #         # Check content type
# #         if 'image' not in response.headers['Content-Type']:
# #             app.logger.error(f"Invalid content type: {response.headers['Content-Type']}")
# #             return jsonify({'error': 'The URL does not point to an image file'}), 400

# #         image = Image.open(BytesIO(response.content))
# #     except requests.RequestException as e:
# #         app.logger.error(f"Error fetching image: {e}")
# #         return jsonify({'error': 'Error fetching image'}), 400
# #     except Exception as e:
# #         app.logger.error(f"Error processing image: {e}")
# #         return jsonify({'error': 'Error processing image'}), 400

#     # Remaining code unchanged


# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     data = request.get_json()
# #     if 'url' not in data:
# #         return jsonify({'error': 'No URL provided'}), 400

# #     image_url = data['url']

# #     try:
# #         response = requests.get(image_url)
# #         response.raise_for_status()  # Ensure we notice bad responses
# #         image = Image.open(BytesIO(response.content))
# #     except requests.RequestException as e:
# #         app.logger.error(f"Error fetching image: {e}")
# #         return jsonify({'error': 'Error fetching image'}), 400
# #     except Exception as e:
# #         app.logger.error(f"Error processing image: {e}")
# #         return jsonify({'error': 'Error processing image'}), 400

# #     try:
# #         app.logger.info("Preprocessing image...")
# #         image = preprocess_image(image)
# #         app.logger.info(f"Image preprocessed successfully. Shape: {image.shape}, Type: {image.dtype}")

# #         app.logger.info("Making prediction...")
# #         predictions = model.predict(image, verbose=0)
# #         app.logger.info(f"Prediction made successfully. Predictions: {predictions}")

# #         predicted_class_index = np.argmax(predictions)
# #         app.logger.info(f"Predicted class index: {predicted_class_index}")

# #         predicted_class_name = class_names[predicted_class_index]
# #         app.logger.info(f"Predicted class name: {predicted_class_name}")

# #         probability = float(np.max(predictions))
# #         app.logger.info(f"Probability: {probability}")

# #         second_predicted_class_index = np.argsort(predictions[0])[-2]
# #         app.logger.info(f"Second predicted class index: {second_predicted_class_index}")

# #         second_predicted_class_name = class_names[second_predicted_class_index]
# #         app.logger.info(f"Second predicted class name: {second_predicted_class_name}")

# #         second_probability = float(predictions[0][second_predicted_class_index])
# #         app.logger.info(f"Second probability: {second_probability}")

# #     except Exception as e:
# #         app.logger.error(f"Error during prediction: {str(e).encode('utf-8', 'ignore').decode('utf-8')}")
# #         return jsonify({'error loc 2': 'Error during prediction'}), 500

# #     return jsonify([
# #         {
# #             'class': predicted_class_name,
# #             'probability': probability
# #         },
# #         {
# #             'class': second_predicted_class_name,
# #             'probability': second_probability
# #         }
# #     ])

# if __name__ == '__main__':
#     app.run(debug=True)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss


from flask import Flask, request, jsonify, send_from_directory, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import logging
import sys
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# Load model
model = tf.keras.models.load_model('../covid19_model.h5')

# Load class names
with open(r'C:\\lung-disease-detection-master\\lung-disease-detection-master\\classes.txt', 'r', encoding='utf-8') as f:
    class_names = f.read().splitlines()

def preprocess_image(image):
    try:
        image = image.resize((64, 64))  # Resize to match your model's input shape
        image = np.array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize if necessary
        return image
    except Exception as e:
        app.logger.error(f"Error during image preprocessing: {e}")
        raise

@app.route('/')
def home():
    return "Welcome to the Lung Disease Detection API."

@app.route('/frontend')
def frontend():
    return send_from_directory('.', 'static/index.html')  # Serve the HTML file from static directory

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     url = data.get('url')
#     if not url:
#         return jsonify({"error": "URL is missing"}), 400

#     # Log that the request has been received
#     app.logger.info(f"Received URL: {url}")
    
#     # Fetch the image from the URL
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Ensure we notice bad responses
#         image = Image.open(BytesIO(response.content))
#     except requests.RequestException as e:
#         app.logger.error(f"Error fetching image: {e}")
#         return jsonify({'error': 'Error fetching image'}), 400
#     except Exception as e:
#         app.logger.error(f"Error processing image: {e}")
#         return jsonify({'error': 'Error processing image'}), 400

#     # Preprocess the image
#     try:
#         app.logger.info("Preprocessing image...")
#         image = preprocess_image(image)
#         app.logger.info(f"Image preprocessed successfully. Shape: {image.shape}, Type: {image.dtype}")

#         # Make prediction
#         app.logger.info("Making prediction...")
#         predictions = model.predict(image, verbose=0)
#         app.logger.info(f"Prediction made successfully. Predictions: {predictions}")

#         # Get the top predicted class
#         predicted_class_index = np.argmax(predictions)
#         predicted_class_name = class_names[predicted_class_index]
#         probability = float(np.max(predictions))

#         # Get the second predicted class
#         second_predicted_class_index = np.argsort(predictions[0])[-2]
#         second_predicted_class_name = class_names[second_predicted_class_index]
#         second_probability = float(predictions[0][second_predicted_class_index])

#         return jsonify([
#             {
#                 'class': predicted_class_name,
#                 'probability': probability
#             },
#             {
#                 'class': second_predicted_class_name,
#                 'probability': second_probability
#             }
#         ])
#     except Exception as e:
#         app.logger.error(f"Error during prediction: {str(e)}")
#         return jsonify({'error': 'Error during prediction'}), 500
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     url = data.get('url')
#     if not url:
#         return jsonify({"error": "URL is missing"}), 400

#     # Log that the request has been received
#     app.logger.info(f"Received URL: {url}")
    
#     # Fetch the image from the URL
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Ensure we notice bad responses

#         # Log the content type of the image
#         app.logger.info(f"Image content type: {response.headers['Content-Type']}")

#         # Ensure the content is of image type
#         if 'image' not in response.headers['Content-Type']:
#             return jsonify({'error': 'The URL does not point to an image file'}), 400

#         image = Image.open(BytesIO(response.content))
#     except requests.RequestException as e:
#         app.logger.error(f"Error fetching image: {e}")
#         return jsonify({'error': 'Error fetching image'}), 400
#     except Exception as e:
#         app.logger.error(f"Error processing image: {e}")
#         return jsonify({'error': 'Error processing image'}), 400

#     # Proceed with image preprocessing and prediction logic...

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Open the uploaded image
        image = Image.open(file)
        app.logger.info(f"Image received: {file.filename}")
        
        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make a prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]
        probability = float(np.max(predictions))

        return jsonify({
            'class': predicted_class_name,
            'probability': probability
        })
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
y_true = np.array([0, 1, 1, 0, 1])  # True labels
y_pred = np.array([0, 1, 0, 0, 1])  # Predicted labels
y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.6, 0.4], [0.4, 0.6]])  
@app.route('/evaluationmetrics')
def evaluation_metrics():
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()  # Convert to list for JSON response
    loss = log_loss(y_true, y_prob)

    # Prepare data to send to frontend
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "log_loss": loss
    }
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True)


