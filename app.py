import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Set page configuration
st.set_page_config(page_title="Spam Email Classifier", page_icon="✉️", layout="centered")

# Title and description
st.title("✉️ Spam Email Classifier")
st.write("This application uses Machine Learning to classify incoming emails or messages as **Spam** or **Ham (Legitimate)**.")

@st.cache_resource
def load_classifier_and_vectorizer():
    # Fetch and prepare dataset dynamically for demo purposes
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X = vectorizer.fit_transform(df['message'])
    y = df['label_num']
    
    # Train SVM (best performer)
    svm_model = SVC(kernel='linear', C=1.0, probability=True)
    svm_model.fit(X, y)
    return vectorizer, svm_model

try:
    vectorizer, model = load_classifier_and_vectorizer()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the classifier: {e}")

# User input
user_input = st.text_area("Paste the email or message content below:", height=150,
                          placeholder="Enter your message here...")

if st.button("Classify Message", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Vectorize input
        transformed_input = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(transformed_input)[0]
        probabilities = model.predict_proba(transformed_input)[0]
        
        # Display Results
        st.markdown("--- ")
        if prediction == 1:
            st.error(f"🚨 **Result: SPAM** (Confidence: {probabilities[1]*100:.2f}%)")
        else:
            st.success(f"✅ **Result: HAM** (Confidence: {probabilities[0]*100:.2f}%)")
