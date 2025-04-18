import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalAveragePooling1D, TextVectorization
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import os

# Load and clean data
def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    if 'description' not in data.columns or 'categories' not in data.columns:
        raise ValueError("Dataset must contain 'description' and 'categories'")
    data['description'] = data['description'].str.lower().fillna('')
    return data

# Encode labels
def encode_labels(y_train, y_test):
    label_encoder = LabelEncoder()
    all_labels = pd.concat([y_train, y_test])
    label_encoder.fit(all_labels)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    return label_encoder, y_train_encoded, y_test_encoded

# Create model
def create_model(vocab_size, label_count):
    model = Sequential([
        Embedding(input_dim=vocab_size + 1, output_dim=128, input_length=500, mask_zero=True),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dense(label_count, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Save vectorizer properly
class VectorizerModule(tf.Module):
    def __init__(self, vectorizer):
        super().__init__()
        self.vectorizer = vectorizer

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def serve(self, x):
        return {'output_0': self.vectorizer(x)}

# Run full pipeline
try:
    # Load or create data
    data_path = 'data.csv'
    if not os.path.exists(data_path):
        data = pd.DataFrame({
            'description': [
                'sci-fi book about space travel',
                'romance novel with happy ending',
                'mystery thriller with detective',
                'fantasy book with dragons'
            ],
            'categories': ['sci-fi', 'romance', 'mystery', 'fantasy']
        })
    else:
        data = load_and_clean_data(data_path)

    # Split
    X = data['description']
    y = data['categories']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_list, X_test_list = X_train.tolist(), X_test.tolist()

    # Vectorization
    vectorizer = TextVectorization(
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=500,
        standardize='lower_and_strip_punctuation'
    )
    vectorizer.adapt(X_train_list)

    # Encode labels
    label_encoder, y_train_encoded, y_test_encoded = encode_labels(y_train, y_test)
    vocab_size = len(vectorizer.get_vocabulary())
    num_classes = len(label_encoder.classes_)

    # Train model
    model = create_model(vocab_size, num_classes)
    model.fit(vectorizer(np.array(X_train_list)), y_train_encoded,
              validation_data=(vectorizer(np.array(X_test_list)), y_test_encoded),
              epochs=10, batch_size=32,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])

    # Save everything
    model.save('book_genre_model.h5')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    vectorizer_module = VectorizerModule(vectorizer)
    tf.saved_model.save(vectorizer_module, "text_vectorizer")

    print("✅ Training complete and files saved!")

except Exception as e:
    import traceback
    print("❌ Error during training:")
    traceback.print_exc()
