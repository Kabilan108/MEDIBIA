from flask import Flask, request, jsonify
from predictions import classify_skin
from classifyDiseases import getsymptomList, develop_inputList, getInfo
import pickle
import json

with open('./models/disease-prediction/my_model_for_healthcare.pkl', 'rb') as f:
    disease_model = pickle.load(f)

app = Flask(__name__)

@app.route('/skin-condition', methods=['POST'])
def classify_skin_condition():
    if 'image' not in request.files:
        return 'No image found in request', 400
    
    image = request.files['image']

    prediction = classify_skin(image)

    return prediction, 200

@app.route('/disease_classifier', methods=['POST'])
def classify_disease():
    request_data = request.get_data()
    request_data = json.loads(request_data.decode('utf-8'))
    userMessage = request_data['message'] 
    message = getsymptomList(userMessage)
    inputList = develop_inputList(message)
    disease = disease_model.predict([inputList])
    info_on_disease = getInfo(disease)
    return (f'I think you might have, {disease}, here is some info about it: \n {info_on_disease}'), 200



@app.route('/')
def index():
    return "<h4>Welcome to MEDIBIA!</h4>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)

