import os
import re
import json
import streamlit as st
import requests
from pymongo import MongoClient
import bcrypt
import logging
import datetime
from bson.objectid import ObjectId

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths and constants
FLASK_API_URL = "http://127.0.0.1:5000"
DB_CONNECTION_STRING = "mongodb+srv://anshgaigawali:anshtini@cluster2.l7iru.mongodb.net/animechatbot?retryWrites=true&w=majority&appName=Cluster2"

# MongoDB connection
client = MongoClient(DB_CONNECTION_STRING)
db = client['animechatbot']
users_collection = db['users']

def signup(email, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user = {"email": email, "password": hashed_password}
    result = users_collection.insert_one(user)
    logger.info(f"User inserted with id: {result.inserted_id}")
    st.success(f"Account created for {email}")

def login(email, password):
    user = users_collection.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode('utf-8'), user["password"]):
        st.success(f"Logged in as {email}")
        return str(user["_id"])
    st.error("Invalid credentials")
    return None

def logout():
    st.session_state["user_id"] = None
    st.success("You have been logged out successfully.")

def delete_account(user_id):
    result = users_collection.delete_one({"_id": ObjectId(user_id)})
    if result.deleted_count > 0:
        st.success("Account deleted successfully.")
        st.session_state["user_id"] = None
    else:
        st.error("Failed to delete the account. Please try again.")

def chatbot(input_text, user_id=None):
    url = f"{FLASK_API_URL}/chat"
    payload = {
        "input": input_text,
        "user_id": user_id
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        response_json = response.json()
        if "error" in response_json:
            st.error(f"Server error: {response_json['error']}")
            return ""
        return response_json.get("response", "")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return ""
    except ValueError as e:
        st.error(f"Error parsing JSON response: {e}")
        return ""

def authentication_page():
    st.header("Anime Chatbot Authentication")

    auth_mode = st.radio("Choose an option", ["Sign In", "Sign Up", "Logout"])

    if auth_mode == "Sign In":
        st.markdown("<h2 class='stSubheader'>Sign In</h2>", unsafe_allow_html=True)
        with st.form(key='sign_in_form'):
            email = st.text_input("Email")
            password = st.text_input("Password", type='password')
            submit_button = st.form_submit_button(label='Sign In')
            if submit_button:
                user_id = login(email, password)
                if user_id:
                    st.session_state["user_id"] = user_id
                
    elif auth_mode == "Sign Up":
        st.markdown("<h2 class='stSubheader'>Sign Up</h2>", unsafe_allow_html=True)
        with st.form(key='sign_up_form'):
            email = st.text_input("Email")
            password = st.text_input("Password", type='password')
            submit_button = st.form_submit_button(label='Sign Up')
            if submit_button:
                signup(email, password)
            
    elif auth_mode == "Logout":
        st.markdown("<h2 class='stSubheader'>Logout</h2>", unsafe_allow_html=True)
        if st.session_state["user_id"]:
            if st.button("Confirm Logout"):
                logout()
        else:
            st.warning("You need to log in to log out.")

def main():
    st.markdown("<h1 class='stHeader'>Anime Chatbot with NLP and Streamlit</h1>", unsafe_allow_html=True)
    menu = ["Home", "Authentication", "Conversation History", "Delete History", "Delete Account", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None

    if choice == "Authentication":
        authentication_page()

    if choice == "Home":
        if st.session_state["user_id"]:
            st.header("Welcome to the Anime Chatbot. Write the name of the anime you want to know about.")
            user_input = st.text_input("You:")
            if user_input:
                st.text_area("Chatbot:", value=chatbot(user_input, st.session_state["user_id"]), height=200)
        else:
            st.warning("You need to log in to chat with the bot.")
            st.info("Please go to the Authentication section.")

    elif choice == "Conversation History":
        if st.session_state["user_id"]:
            st.header("Conversation History")
            user_doc = users_collection.find_one({"_id": ObjectId(st.session_state["user_id"])})
            if user_doc and "history" in user_doc and user_doc["history"]:
                user_history = user_doc["history"]
                for history in user_history:
                    st.text(f"User: {history['user_input']}\nChatbot: {history['response']}\nTimestamp: {history['timestamp']}")
                    st.markdown("---")
            else:
                st.warning("No conversation history found for this user.") # Add this line to show a warning
        else:
            st.warning("You need to log in to see conversation history.")

    elif choice == "Delete History":
        if st.session_state["user_id"]:
            st.header("Delete Conversation History")
            user_doc = users_collection.find_one({"_id": ObjectId(st.session_state["user_id"])})
            if user_doc and "history" in user_doc and user_doc["history"]:
                if st.button("Delete history"):
                    users_collection.update_one({"_id": ObjectId(st.session_state["user_id"])}, {"$set": {"history": []}})
                    st.success("Conversation history deleted for the current user.")
            else:
                st.warning("No conversation history to delete.")
        else:
            st.warning("You need to log in to delete history.")

    elif choice == "Delete Account":
        if st.session_state["user_id"]:
            st.header("Delete Account")
            if st.button("Confirm Delete Account"):
                delete_account(st.session_state["user_id"])
        else:
            st.warning("You need to log in to delete your account.")

    elif choice == "About":
        st.header("About")
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
