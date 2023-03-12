"""
Function for performing image predictions
"""

# Imports
import tensorflow as tf


# Load the trained model
model = tf.keras.models.load_model('models/dermanet-latest.h5')
class_names = [
    'Acne', 'Actinic Keratosis (Malignant)', 'Atopic Dermatitis', 
    'Bullous Disease', 'Cellulitis Impetigo', 'Eczema', 
    'Exanthems (Drug Eruptions)', 'Alopecia', 'Herpes or HPV',
    'Pigmentation Disorder', 'Lupus', 'Melanoma', 'Nail Fungus', 'Poison Ivy',
    'Psoriasis', 'Lyme Disease', 'Benign Tumors', 'Systemic Disease',
    'Ringworm', 'Hives', 'Vascular Tumors', 'Vasculitis', 'Warts Molluscum'
]


def classify(image):
    # Assuming that image is from Flask's request.files object

    # Preprocess the image
    image = tf.image.decode_image(image.read(), channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.vgg16.preprocess_input(image)

    # Perform inference
    prediction = model.predict(tf.expand_dims(image, axis=0))

    # Postprocess the prediction
    predicted_class = class_names[tf.argmax(prediction[0], axis=-1)]
    confidence = tf.nn.softmax(prediction[0])[tf.argmax(prediction[0], axis=-1)]
    

    return f'{predicted_class} with {confidence*100:.2f}% confidence'

