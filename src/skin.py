"""
Function for performing image predictions
"""

# Imports
import tensorflow as tf


# Load the trained model
model = tf.keras.models.load_model('models/dermanet-latest.h5')
class_names = []

def classify(image):
    # Assuming that image is from Flask's request.files object

    # Preprocess the image
    image = tf.image.decode_image(image.read(), channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.vgg16.preprocess_input(image)

    # Perform inference
    prediction = model.predict(tf.expand_dims(image, axis=0))

    # Postprocess the prediction
    I = tf.argmax(prediction[0])
    predicted_class = class_names[I]
    confidence = prediction[0][I]

    return f'{predicted_class} with {confidence*100:.2f}% confidence'

