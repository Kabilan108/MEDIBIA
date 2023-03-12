from flask import Flask, request, jsonify
from predictions import classify_skin

app = Flask(__name__)
    

@app.route('/skin-condition', methods=['POST'])
def classify_skin_condition():
    if 'image' not in request.files:
        return 'No image found in request', 400
    
    image = request.files['image']

    prediction = classify_skin(image)

    return prediction, 200


@app.route('/')
def index():
    return "<h4>Welcome to MEDIBIA!</h4>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
