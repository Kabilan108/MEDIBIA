"""
Function for performing image predictions
"""

# Imports
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
import numpy as np
import cv2

# Load the trained models
VGG19 = tf.keras.models.load_model('models/skin/VGG19-final.h5')
dermanet = tf.keras.models.load_model('models/skin/dermanet-final.h5')

# Define the class names
class_names = [
    'Acne', 'Actinic Keratosis (Malignant)', 'Atopic Dermatitis', 
    'Bullous Disease', 'Cellulitis Impetigo', 'Eczema', 
    'Exanthems (Drug Eruptions)', 'Alopecia', 'Herpes or HPV',
    'Pigmentation Disorder', 'Lupus', 'Melanoma', 'Nail Fungus', 'Poison Ivy',
    'Psoriasis', 'Lyme Disease', 'Benign Tumors', 'Systemic Disease',
    'Ringworm', 'Hives', 'Vascular Tumors', 'Vasculitis', 'Warts Molluscum'
]


def classify_skin(img):
    """Classify an image of a skin condition"""

    # Load image
    images = []
    img = cv2.resize(img, (100,100))
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

