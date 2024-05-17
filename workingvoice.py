import streamlit as st
import speech_recognition as sr
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from gtts import gTTS
import os

# Load the intents file
with open("finalintent.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

train_data = pd.read_csv('t2.csv')
X_train = train_data.drop('prognosis', axis=1)
y_train = train_data['prognosis']

model1 = RandomForestClassifier()
model1.fit(X_train, y_train)

def collect_symptoms(symptoms_input):
    symptoms_list = symptoms_input.lower().strip().split(',')
    symptoms_vector = np.zeros(len(X_train.columns))
    for symptom in symptoms_list:
        symptom = symptom.strip()
        if symptom in X_train.columns:
            symptoms_vector[X_train.columns.get_loc(symptom)] = 1
    return symptoms_vector

def chat(inp):
    results = model.predict([bag_of_words(inp, words)])[0]
    results_index = np.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.5:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                chatvoice = random.choice(responses)
                return chatvoice
    else:
        return "I didn't get that, try again."

def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"

    return response

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        font-size: 32px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 20px 48px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 32px;
        margin: 8px 4px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: white;
        color: black;
        border: 4px solid #4CAF50;
    }
    .stTextInput>div>input {
        border: 4px solid #4CAF50;
        border-radius: 8px;
        padding: 20px;
        font-size: 32px;
    }
    .title {
        color: #4CAF50;
        text-align: center;
        font-size: 48px;
        font-weight: bold;
    }
    .subtitle {
        color: #333;
        text-align: center;
        font-size: 40px;
        margin-top: -20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>👨🏻‍⚕️🩺 Medical Diagnosis and First Aid Chatbot 🩺👨🏻‍⚕️</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='subtitle'>How can we assist you today?</h2>", unsafe_allow_html=True)

if 'init' not in st.session_state:
    cvoice = "Do you want 'diagnosis' or 'first aid' help? Speak your choice: "
    st.write(cvoice)
    myobj = gTTS(text=cvoice, lang='en', slow=False)
    myobj.save("cvoice.mp3")
    os.system("afplay cvoice.mp3")
    st.session_state.init = True

recognizer = sr.Recognizer()
microphone = sr.Microphone()

if 'choice' not in st.session_state:
    st.session_state.choice = ""

choice = st.session_state.choice

if choice == "":
    st.write("Click on the button below to speak your choice")

    if st.button("Speak your choice"):
        response = recognize_speech_from_mic(recognizer, microphone)
        if response["success"]:
            if response["transcription"]:
                choice = response["transcription"].lower().strip()
                st.session_state.choice = choice
                st.write(f"You said: {choice}")
            else:
                st.write("I didn't catch that. Please try again.")
        else:
            st.write("Error: " + response["error"])

if choice == 'diagnosis':
    symptoms_input = st.text_input("Enter your symptoms separated by commas:")
    if st.button("Diagnose"):
        symptoms_vector = collect_symptoms(symptoms_input)
        prognosis = model1.predict([symptoms_vector])[0]
        diagvoice = f"Based on your symptoms, the diagnosis is: {prognosis}. Please enter prognosis in chat below."
        st.write(diagvoice) 
        tts = gTTS(text=diagvoice, lang='en', slow=False)
        tts.save("diagnosis.mp3")
        os.system("afplay diagnosis.mp3")
        st.session_state.diagnosis = True

    if 'diagnosis' in st.session_state:
        if st.button("Speak"):
            response = recognize_speech_from_mic(recognizer, microphone)
            if response["success"]:
                if response["transcription"]:
                    inp = response["transcription"]
                    st.write(f"You said: {inp}")
                    response = chat(inp)
                    st.write(response)  
                    tts = gTTS(text=response, lang='en', slow=False)
                    tts.save("response.mp3")
                    os.system("afplay response.mp3")
                else:
                    st.write("I didn't catch that. Please try again.")
            else:
                st.write("Error: " + response["error"])

elif choice == 'first aid':
    if 'first_aid_init' not in st.session_state:
        st.write("Please speak your problem.")
        myobj = gTTS(text="Please speak your problem.", lang='en', slow=False)
        myobj.save("first_aid.mp3")
        os.system("afplay first_aid.mp3")
        st.session_state.first_aid_init = True

    if st.button("Speak your problem"):
        response = recognize_speech_from_mic(recognizer, microphone)
        if response["success"]:
            if response["transcription"]:
                inp = response["transcription"]
                st.write(f"You said: {inp}")
                response = chat(inp)
                st.write(response) 
                tts = gTTS(text=response, lang='en', slow=False)
                tts.save("response.mp3")
                os.system("afplay response.mp3")
            else:
                st.write("I didn't catch that. Please try again.")
        else:
            st.write("Error: " + response["error"])

else:
    if choice:
        st.write("Invalid choice, please select 'diagnosis' or 'first aid'.")

# Reset the state for choice if the user wants to start over
if st.button("Reset"):
    st.session_state.choice = ""
    st.session_state.diagnosis = False
    st.session_state.first_aid_init = False
    st.write("Choice reset. Please speak your choice again.")
