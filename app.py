from flask import Flask, request, jsonify
import joblib
import os
import re
from sklearn.metrics.pairwise import cosine_similarity
import json
import datetime
from pymongo import MongoClient
import logging
from bson.objectid import ObjectId
import nltk

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# Paths and constants
MODEL_PATH = 'C:/Users/USER/Desktop/Project/ChatBot/chatbot_model1.pkl'
VECTORIZER_PATH = 'C:/Users/USER/Desktop/Project/ChatBot/vectorizer1.pkl'
INTENT_FILE_PATH = 'C:/Users/USER/Desktop/Project/ChatBot/top_1000_anime.json'
DB_CONNECTION_STRING = "mongodb+srv://anshgaigawali:anshtini@cluster2.l7iru.mongodb.net/animechatbot?retryWrites=true&w=majority&appName=Cluster2"

# Load NLTK data
nltk.data.path.append(os.path.abspath("nltk_data"))

# Load data function
def load_data():
    try:
        with open(INTENT_FILE_PATH, "r", encoding="utf-8") as file:
            intents = json.load(file)
    except FileNotFoundError:
        logger.error("JSON file not found. Please ensure the file path is correct.")
        return [], [], []
    except json.JSONDecodeError:
        logger.error("Error decoding JSON file.")
        return [], [], []

    tags, patterns, responses = [], [], []
    for intent in intents:
        for pattern in intent['patterns']:
            if pattern:
                tags.append(intent['tag'])
                patterns.append(pattern.lower())
                responses.append(intent['responses'])
    return tags, patterns, responses

# Load training data
tags, patterns, responses = load_data()

# Load model and vectorizer
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    try:
        clf = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception as e:
        logger.error(f"Error loading model/vectorizer: {e}")
else:
    try:
        vectorizer = TfidfVectorizer()
        clf = LogisticRegression(random_state=0, max_iter=10000)
        x = vectorizer.fit_transform(patterns)
        clf.fit(x, tags)
        joblib.dump(clf, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
    except Exception as e:
        logger.error(f"Error training model/vectorizer: {e}")

def normalize_input(input_text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", input_text.lower())

def find_best_response(input_text):
    input_vec = vectorizer.transform([normalize_input(input_text)])
    max_similarity_index = cosine_similarity(input_vec, vectorizer.transform(patterns)).flatten().argmax()
    return "\n".join(responses[max_similarity_index])

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        logger.debug(f"Received data: {data}")
        
        if not data:
            error_msg = 'No input data provided'
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
        
        user_input = data['input']
        user_id = data.get('user_id', None)
        
        response_text = find_best_response(user_input)
        if user_id:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            client = MongoClient(DB_CONNECTION_STRING)
            db = client['animechatbot']
            db.users.update_one({"_id": ObjectId(user_id)}, {"$push": {"history": {"user_input": user_input, "response": response_text, "timestamp": timestamp}}})
        
        logger.debug(f"Response text: {response_text}")
        return jsonify({'response': response_text})
    except Exception as e:
        logger.exception("Exception occurred")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run()
