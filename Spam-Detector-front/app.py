import pickle
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, render_template
import string
import re
import joblib
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)


# 加载 BiLSTM CNN DT
model_bilstm = tf.keras.models.load_model('save_model/bilstm/LSTM.h5')
model_cnn = tf.keras.models.load_model('save_model/cnn/CNN.h5')
model_dt = joblib.load('save_model/dt/DT.pkl')

# 加载 NB
nb_classifier_loaded = joblib.load('save_model/nb/nb_classifier.pkl')
vectorizer_loaded = joblib.load('save_model/nb/tfidf_vectorizer.pkl')
label_encoder_loaded = joblib.load('save_model/nb/label_encoder.pkl')

max_features = 5000

with open('save_model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    print("tokenizer!")

label_encoder = LabelEncoder()

ENGLISH_STOP_WORDS = set(stopwords.words('english'))


def remove_special_characters(word):
    return word.translate(str.maketrans('', '', string.punctuation))


def remove_stop_words(words):
    return [word for word in words if word not in ENGLISH_STOP_WORDS]


def preprocess_text(text):
    text = text.lower()
    text = remove_special_characters(text)
    words = word_tokenize(text)
    words = remove_stop_words(words)
    text = ' '.join(words)
    return text


def predict_email_category(email_content):
    processed_text = preprocess_text(email_content)
    # print(processed_text)
    email_seq = tokenizer.texts_to_sequences([processed_text])
    # print(email_seq)
    email_padded = pad_sequences(email_seq, maxlen=500, padding='post')
    # print(email_padded)

    # CNN BiLSTM 预测
    prediction_cnn = model_cnn.predict(email_padded)[0][0]
    prediction_bilstm = model_bilstm.predict(email_padded)[0][0]
    print(prediction_cnn)
    print(prediction_bilstm)

    # DT 预测
    prediction_dt = model_dt.predict_proba(email_padded)
    print(prediction_dt)
    prediction_dt = prediction_dt[0][1]

    # NB 预测
    message_tfidf = vectorizer_loaded.transform([processed_text])
    prediction_nb = nb_classifier_loaded.predict_proba(message_tfidf)
    print(prediction_nb)
    prediction_nb = prediction_nb[0][1]

    result = {
        "CNN": {"Probability": float(prediction_cnn),
                "Prediction": "Spam" if prediction_cnn > 0.5 else "Legitimate"},
        "BiLSTM": {"Probability": float(prediction_bilstm),
                   "Prediction": "Spam" if prediction_bilstm > 0.5 else "Legitimate"},
        "DecisionTree": {"Probability": float(prediction_dt),
                         "Prediction": "Spam" if prediction_dt > 0.5 else "Legitimate"},
        "NaiveBayes": {"Probability": float(prediction_nb),
                       "Prediction": "Spam" if prediction_nb > 0.5 else "Legitimate"}
    }

    return result


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    email_content = request.json.get('email_content')

    # Get prediction
    result = predict_email_category(email_content)
    print(result)

    # Return result as JSON
    return jsonify({'result': result})


# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=8080)
