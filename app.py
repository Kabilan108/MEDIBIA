from flask import Flask, request, jsonify
from predictions import classify_skin
from classifyDiseases import getsymptomList, encode_symptoms, getInfo
import pickle
import json

with open('./models/disease-prediction/MultinomialBayes.pkl', 'rb') as f:
    disease_model = pickle.load(f)

app = Flask(__name__)

@app.route('/skin-condition', methods=['POST'])
def classify_skin_condition():
    if 'image' not in request.files:
        return 'No image found in request', 400
    
    image = request.files['image']

    prediction = classify_skin(image)

    return {'message': prediction}, 200


@app.route('/disease-classifier', methods=['POST'])
def classify_disease():
    request_data = request.get_data()
    request_data = json.loads(request_data.decode('utf-8'))
    userMessage = request_data['message'] 
    message = getsymptomList(userMessage)
    symptoms = encode_symptoms(message)
    disease = disease_model.predict(symptoms)
    info_on_disease = getInfo(disease)
    
    res = f'I think you might have, {disease[0]}, here is some info about it:\n{info_on_disease}'
    return {'message': res}, 200


@app.route('/')
def index():
    return "<h4>Welcome to MEDIBIA!</h4>"


if __name__ == '__main__':
    app.run(threaded=True, port=5000)

