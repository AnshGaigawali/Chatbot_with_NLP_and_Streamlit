import os
import json
import datetime
import csv
import re
import streamlit as st
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import nltk

# Handle SSL issues
ssl._create_default_https_context = ssl._create_unverified_context

# download NLTK data
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# load intents from JSON
def load_data():
    file_path = os.path.abspath("C:/Users/USER/Desktop/Project/ChatBot/top_1000_anime.json")
    try:
        print("Loading data from JSON file...")
        with open(file_path, "r", encoding="utf-8") as file:
            intents = json.load(file)
        print("Data loaded successfully.")
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
            if pattern:  # Ensure patterns are not empty
                tags.append(intent['tag'])
                patterns.append(pattern.lower())  # Convert to lowercase
                responses.append(intent['responses'])  # Collect all responses

    print("Data processed successfully.")
    return tags, patterns, responses

# define paths for saved model and vectorizer
model_path = 'C:/Users/USER/Desktop/Project/ChatBot/chatbot_model1.pkl'
vectorizer_path = 'C:/Users/USER/Desktop/Project/ChatBot/vectorizer1.pkl'

# initialize variables to avoid "missing pattern" previous bug
tags, patterns, responses = load_data()

# check if the model and vectorizer exist or train and save them
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    try:
        print("Loading existing model and vectorizer...")
        clf = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Model and vectorizer loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {e}")
else:
    try:
        print("Training new model and vectorizer...")
        vectorizer = TfidfVectorizer()
        clf = LogisticRegression(random_state=0, max_iter=10000)

        x = vectorizer.fit_transform(patterns)
        y = tags
        clf.fit(x, y)

        joblib.dump(clf, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print("Model and vectorizer trained and saved successfully.")
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

def chatbot(input_text):
    response = find_best_response(input_text)
    return response

def main():
    st.title("Anime Chatbot with NLP and Streamlit")

    # sidebar menu
    menu = ["Home", "Conversation History", "Delete History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    st.sidebar.markdown("---")

    # file path for chat log
    log_file_path = os.path.abspath("C:/Users/USER/Desktop/Project/ChatBot/chat_log.csv")

    if not os.path.exists(log_file_path):
        print("Creating chat log file...")
        with open(log_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
        print("Chat log file created successfully.")

    if choice == "Home":
        st.subheader("Welcome to the Anime Chatbot. Write the name of the anime you want to know about.")
        user_input = st.text_input("You:")
        response = ""

        if user_input:
            with st.spinner("Processing..."):
                response = chatbot(user_input)
                print("User input processed.")

            st.success("Received your input! Here's the response:")
            st.text_area("Chatbot:", value=response, height=200, max_chars=None)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Logging conversation at {timestamp}...")
            with open(log_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])
            print("Conversation logged successfully.")

    elif choice == "Conversation History":
        st.header("Conversation History")
        print("Displaying conversation history...")
        with open(log_file_path, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                st.text(f"User: {row[0]}\nChatbot: {row[1]}\nTimestamp: {row[2]}")
                st.markdown("---")
        print("Conversation history displayed.")

    elif choice == "Delete History":
        st.header("Delete Conversation History")
        if st.button("Delete history"):
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
                st.success("Conversation history deleted")
                print("Conversation history deleted.")
            else:
                st.warning("No conversation history found")
                print("No conversation history found.")

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

        print("Displayed information about the project.")
        
if __name__ == '__main__':
    main()
