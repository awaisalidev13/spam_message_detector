import streamlit as st
import joblib
import re

# Cache model and vectorizer loading so they don't reload on every interaction
@st.cache_resource
def load_artifacts():
    model = joblib.load("D:/spam_message_detector/model.joblib")
    vectorizer = joblib.load("D:/spam_message_detector/vectorizer.joblib")
    return model, vectorizer

model, tfid = load_artifacts()

st.title("Spam Message Detector")
st.write("Enter a message below to classify it as Spam or Ham.")

message = st.text_input("Enter a message:")

if st.button("Check Message"):
    if message.strip():
        # Apply the same preprocessing as in training
        clean_msg = message.lower()
        clean_msg = re.sub(r'[^a-zA-Z0-9\s.]', '', clean_msg)

        # Transform using the pre-fitted vectorizer (DO NOT use fit_transform)
        message_vector = tfid.transform([clean_msg])

        # Predict
        prediction = model.predict(message_vector)[0]

        # Notebook mapping: 'ham': 1, 'spam': 0
        if prediction == 0:
            st.error("This is a spam message.")
        else:
            st.success("This is not a spam message (Ham).")
    else:
        st.warning("Please enter a message first.")