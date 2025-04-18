from django.shortcuts import render
import tensorflow as tf
import joblib
import os
import numpy as np

# Define base directory and paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'book_genre_model.h5')
vectorizer_path = os.path.join(BASE_DIR, 'text_vectorizer')
label_encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')

# Load once on startup
model = tf.keras.models.load_model(model_path)
vectorizer = tf.saved_model.load(vectorizer_path)
label_encoder = joblib.load(label_encoder_path)

def classify(request):
    prediction = None
    description = ""
    error_message = None
    confidence = None

    if request.method == 'POST':
        description = request.POST.get('description', '').strip().lower()
        try:
            input_tensor = tf.constant([description])
            vectorized = vectorizer.serve(input_tensor)['output_0']
            pred_probs = model.predict(vectorized)
            pred_index = np.argmax(pred_probs, axis=1)[0]
            prediction = label_encoder.inverse_transform([pred_index])[0]
            confidence = float(pred_probs[0][pred_index]) * 100
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_message = f"Error during prediction: {str(e)}"

    return render(request, 'classifier/index.html', {
        'prediction': prediction,
        'description': description,
        'confidence': confidence,
        'error': error_message
    })
