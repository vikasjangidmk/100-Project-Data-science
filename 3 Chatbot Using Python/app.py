import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Creating the intents for the chatbot
intents = [
    {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "Howdy", "Good morning", "Good evening", "What's up?", "Greetings!"],
     "responses": ["Hello! How can I assist you today?", "Hi there! How can I help?", "Hey! What can I do for you?"]},
    {"tag": "goodbye", "patterns": ["Bye", "Goodbye", "See you later", "Take care", "Catch you later", "I'm leaving"],
     "responses": ["Goodbye! Have a great day!", "See you later!", "Take care and stay safe!"]},
    {"tag": "thanks", "patterns": ["Thank you", "Thanks", "Thanks a lot", "Much appreciated", "I'm grateful"],
     "responses": ["You're welcome!", "No problem at all!", "Happy to help!"]},
    {"tag": "help", "patterns": ["Can you help me?", "I need assistance", "Help me please", "What support do you provide?"],
     "responses": ["Sure, I'm here to assist. What's the issue?", "Of course! Tell me what you need help with.", "How can I support you today?"]},
    {"tag": "weather", "patterns": ["What's the weather like?", "How's the weather?", "Is it sunny today?", "Will it rain today?"],
     "responses": ["I can't provide real-time weather updates, but you can check a weather app or website.", "I'm sorry, I don't have live weather data."]},
    {"tag": "about", "patterns": ["Who are you?", "What are you?", "Tell me about yourself", "What can you do?"],
     "responses": ["I am a chatbot here to assist you with your queries.", "I'm a virtual assistant designed to help with various tasks."]},
    {"tag": "funny", "patterns": ["Tell me a joke", "Make me laugh", "Do you know any jokes?", "Be funny"],
     "responses": ["Why don't scientists trust atoms? Because they make up everything!", "What do you call fake spaghetti? An impasta!", "I told my computer I needed a break, and now it won’t stop sending me KitKat ads."]},
    {"tag": "time", "patterns": ["What time is it?", "Can you tell me the time?", "What's the current time?"],
     "responses": ["I can't check the time, but your device should have the correct time."]},
    {"tag": "date", "patterns": ["What's the date today?", "Can you tell me today's date?", "What's today's date?"],
     "responses": ["I'm not sure of the exact date, but you can check your calendar.", "Today's date should be displayed on your device."]},
    {"tag": "motivation", "patterns": ["Give me some motivation", "Motivate me", "I need inspiration", "Feeling low"],
     "responses": ["Believe in yourself and all that you are. Know that there is something inside you that is greater than any obstacle.", "You're capable of amazing things! Keep pushing forward.", "Every day is a new opportunity to grow and improve. You've got this!"]},
    {"tag": "programming", "patterns": ["How do I learn programming?", "What's the best way to code?", "Can you help me with programming?"],
     "responses": ["Start with basics like Python or JavaScript. Practice regularly and work on small projects.", "Online platforms like Codecademy, freeCodeCamp, or Coursera are great for learning programming.", "Consistent practice and problem-solving on sites like HackerRank or LeetCode will boost your skills."]},
    {"tag": "general_knowledge", "patterns": ["Tell me something interesting", "Do you know any fun facts?", "Share some knowledge"],
     "responses": ["Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still edible!", "The Eiffel Tower can be 15 cm taller during the summer due to thermal expansion of metal."]},
    {"tag": "travel", "patterns": ["What are the best travel destinations?", "Can you suggest some vacation spots?", "Where should I travel?"],
     "responses": ["Paris is great for romance and art lovers. Bali is ideal for tropical getaways. Iceland is perfect for nature enthusiasts!", "For a cultural experience, visit Kyoto in Japan. For adventure, try New Zealand or Peru!"]},
    {"tag": "health", "patterns": ["How can I stay healthy?", "What are some health tips?", "Any advice for a healthy lifestyle?"],
     "responses": ["Stay hydrated, eat a balanced diet, exercise regularly, and get enough sleep.", "Meditation and mindfulness can also improve your mental health."]},
    {"tag": "technology", "patterns": ["What's the latest in tech?", "Tell me about new technologies", "What's trending in technology?"],
     "responses": ["AI and machine learning are revolutionizing industries. Blockchain is advancing financial systems, and renewable energy technologies are growing rapidly.", "You might want to explore innovations in 5G, electric vehicles, or augmented reality."]},
]
# Initialize vectorizer and classifier
vectorizer = TfidfVectorizer()
classifier = LogisticRegression(random_state=0, max_iter=10000)

# Prepare training data
tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent["tag"])
        patterns.append(pattern)

# Train the classifier
x = vectorizer.fit_transform(patterns)
y = tags
classifier.fit(x, y)

# Chatbot response function
def chatbot_response(text):
    input_text = vectorizer.transform([text])
    predicted_tag = classifier.predict(input_text)[0]
    for intent in intents:
        if intent["tag"] == predicted_tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I don't understand."

# Streamlit App
st.title("Chatbot with Streamlit")
st.write("Welcome to the chatbot! Type your query below and press the 'Run' button.")

# Input field
user_input = st.text_input("You:", key="user_input")

# Run button
if st.button("Run"):
    if user_input:
        response = chatbot_response(user_input)
        st.write(f"Chatbot: {response}")
    else:
        st.write("Chatbot: Please enter a query.")

st.write("Type 'exit' or 'quit' to end the conversation.")
