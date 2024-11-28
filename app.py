from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import traceback

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model using joblib
try:
    model = joblib.load('heart_disease_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


@app.route('/')
def home():
    return "Heart Disease Prediction API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()

        # Ensure 'features' is passed as a list of numerical values
        features = np.array(data['features']).reshape(1, -1)

        # Print features for debugging
        print("Input features:", features)

        # Verify model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Make prediction using the model
        prediction = model.predict(features)

        # Return the result as a JSON response
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        # Detailed error logging
        print(f"Prediction Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)