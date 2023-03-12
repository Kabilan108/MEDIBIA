"""
Function for performing image predictions
"""

# Imports
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the trained models
VGG19 = tf.keras.models.load_model('models/skin/VGG19-final.h5')
dermanet = tf.keras.models.load_model('models/skin/dermanet-final.h5')

# Define the class names
class_names = [
    'Acne', 'Actinic Keratosis (Malignant Lesion)', 'Atopic Dermatitis', 
    'Eczema', 'Nail Fungus', 'Psoriasis'
]


def classify_skin(image):
    """Classify an image of a skin condition"""

    # Load image
    images = []
    img = Image.open(image).convert('RGB')
    img = np.array( img.resize((100, 100)) )
    images.append(img)

    # Feature extraction with VGG19 (Imagenet)
    X = np.asarray(images)
    X = preprocess_input(X)
    features = VGG19.predict(X)

    # Reshape input for classifier
    features = features.reshape(X.shape[0], 4608)

    # Predict the class label
    pred = dermanet.predict(features)
    class_idx = tf.argmax(pred[0], axis=-1)
    pred_class = class_names[ class_idx ]
    confidence = tf.nn.softmax(pred[0])[ class_idx ]

    return f'Theres a {confidence*100:.0f}% chance this is {pred_class}.'
