import os
import json
import datetime
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import nltk
from pymongo import MongoClient
import bcrypt
from dotenv import load_dotenv
import csv
import ssl
import logging
from bson.objectid import ObjectId

# Ensure SSL context for HTTPS requests
ssl._create_default_https_context = ssl._create_unverified_context

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.data.path.append(os.path.abspath("nltk_data"))

# Load intents from JSON
def load_data():
    file_path = os.path.abspath("C:/Users/USER/Desktop/Project/ChatBot/top_1000_anime.json")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            intents = json.load(file)
    except FileNotFoundError:
        st.error("JSON file not found. Please ensure the file path is correct.")
        return [], [], []
    except json.JSONDecodeError:
        st.error("Error decoding JSON file.")
        return [], [], []

    tags = []
    patterns = []
    responses = []
    for intent in intents:
        for pattern in intent['patterns']:
            if pattern:
                tags.append(intent['tag'])
                patterns.append(pattern.lower())
                responses.append(intent['responses'])
    return tags, patterns, responses

# Define paths for saved model and vectorizer
model_path = 'C:/Users/USER/Desktop/Project/ChatBot/chatbot_model1.pkl'
vectorizer_path = 'C:/Users/USER/Desktop/Project/ChatBot/vectorizer1.pkl'
tags, patterns, responses = load_data()

# Check if the model and vectorizer exist, or train and save them
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    try:
        clf = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {e}")
else:
    try:
        vectorizer = TfidfVectorizer()
        clf = LogisticRegression(random_state=0, max_iter=10000)

        x = vectorizer.fit_transform(patterns)
        y = tags
        clf.fit(x, y)

        joblib.dump(clf, model_path)
        joblib.dump(vectorizer, vectorizer_path)
    except Exception as e:
        st.error(f"Error training model/vectorizer: {e}")

def normalize_input(input_text):
    input_text = input_text.lower()
    input_text = re.sub(r"[^a-zA-Z0-9\s]", "", input_text)
    return input_text

def find_best_response(input_text):
    input_text = normalize_input(input_text)
    input_vec = vectorizer.transform([input_text])
    
    similarities = cosine_similarity(input_vec, vectorizer.transform(patterns)).flatten()
    max_similarity_index = similarities.argmax()
    best_response = "\n".join(responses[max_similarity_index])
    return best_response

# Load environment variables
load_dotenv()
connection_string = "mongodb+srv://anshgaigawali:anshtini@cluster2.l7iru.mongodb.net/animechatbot?retryWrites=true&w=majority&appName=Cluster2"

# MongoDB setup
client = MongoClient(connection_string)
db = client['animechatbot']
users_collection = db['users']

def signup(email, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user = {
        "email": email,
        "password": hashed_password
    }
    result = users_collection.insert_one(user)
    logger.info(f"User inserted with id: {result.inserted_id}")
    st.success(f"Account created for {email}")

def login(email, password):
    user = users_collection.find_one({"email": email})
    logger.debug(f"User login: {user}")
    if user and bcrypt.checkpw(password.encode('utf-8'), user["password"]):
        st.success(f"Logged in as {email}")
        return str(user["_id"])
    else:
        st.error("Invalid credentials")
        return None

def chatbot(input_text, user_id=None):
    response = find_best_response(input_text)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file_path = os.path.abspath("C:/Users/USER/Desktop/Project/ChatBot/chat_log.csv")

    if user_id:
        users_collection.update_one(
            {"_id": ObjectId(user_id)},  # Assuming user_id is a valid ObjectId
            {"$push": {"history": {"user_input": input_text, "response": response, "timestamp": timestamp}}}
        )
    else:
        with open(log_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([input_text, response, timestamp])
    return response

def main():
    st.title("Anime Chatbot with NLP and Streamlit")
    menu = ["Home", "Login", "Signup", "Conversation History", "Delete History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    st.sidebar.divider()
    
    log_file_path = os.path.abspath("C:/Users/USER/Desktop/Project/ChatBot/chat_log.csv")

    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None

    if choice == "Signup":
        st.subheader("Create Account")
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')
        if st.button("Signup"):
            signup(email, password)
    
    if choice == "Login":
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            user_id = login(email, password)
            if user_id:
                st.session_state["user_id"] = user_id

    if choice == "Home":
        st.subheader("Welcome to the Anime Chatbot. Write the name of the anime you want to know about.")
        user_input = st.text_input("You:")
        response = ""
        if user_input:
            response = chatbot(user_input, st.session_state["user_id"])
            st.text_area("Chatbot:", value=response, height=200)

    elif choice == "Conversation History":
        st.header("Conversation History")
        if st.session_state["user_id"]:
            user_doc = users_collection.find_one({"_id": ObjectId(st.session_state["user_id"])})
            if user_doc and "history" in user_doc:
                user_history = user_doc["history"]
                for history in user_history:
                    st.text(f"User: {history['user_input']}\nChatbot: {history['response']}\nTimestamp: {history['timestamp']}")
                    st.markdown("---")
            else:
                st.warning("No conversation history found for this user.")
        else:
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r', encoding='utf-8') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    next(csv_reader)
                    for row in csv_reader:
                        st.text(f"User: {row[0]}\nChatbot: {row[1]}\nTimestamp: {row[2]}")
                        st.markdown("---")
            else:
                st.warning("No conversation history found.")

    elif choice == "Delete History":
        st.header("Delete Conversation History")
        if st.button("Delete history"):
            if st.session_state["user_id"]:
                users_collection.update_one(
                    {"_id": ObjectId(st.session_state["user_id"])},
                    {"$set": {"history": []}}
                )
                st.success("Conversation history deleted for the current user.")
            else:
                if os.path.exists(log_file_path):
                    os.remove(log_file_path)
                    with open(log_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
                    st.success("CSV conversation history deleted and recreated.")
                else:
                    st.warning("No conversation history found to delete.")

    elif choice == "About":
        st.write("This project demonstrates an anime-specific chatbot built using NLP techniques.")
        st.subheader("Overview:")
        st.write("""
        1. **Dataset**: The chatbot is trained on a dataset of over 1000 anime titles with various patterns and responses.
        
        2. **Model**: We use Logistic Regression to classify user queries into different anime categories.

        3. **Interface**: The Streamlit interface allows users to interact with the chatbot seamlessly.

        4. **Purpose**: The chatbot helps users find information about their favorite animes including details like title, type, description, and more.
        """)
        st.subheader("Additional Information:")
        st.write("Feel free to explore and ask about different anime titles!")

if __name__ == '__main__':
    main()
