from flask import Flask, request, jsonify
from predictions import classify_skin
import pickle


app = Flask(__name__)

disease_model = pickle.load(open('./notebooks/disease-prediction/my_model_for_healthcare', 'rb'))

@app.route('/skin-condition', methods=['POST'])
def classify_skin_condition():
    if 'image' not in request.files:
        return 'No image found in request', 400
    
    image = request.files['image']

    prediction = classify_skin(image)

    return prediction, 200

@app.route('/disease_classifier', methods=['POST'])
def classify_disease():
    request_data = request.data
    request_data = json.loads(request_data.decode('utf-8'))
    userMessage = request_data['message'] 


@app.route('/')
def index():
    return "<h4>Welcome to MEDIBIA!</h4>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
