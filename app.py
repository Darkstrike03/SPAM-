import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC

# Set page configuration
st.set_page_config(page_title="Spam Email Classifier", page_icon="✉️", layout="centered")

# Title and description
st.title("✉️ Spam Email Classifier")
st.write("This application uses Machine Learning to classify incoming emails or messages as **Spam** or **Ham (Legitimate)**.")

@st.cache_resource
def load_robust_classifier_and_vectorizer():
    # Fetch and prepare dataset dynamically for demo purposes
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    # Use our robust Word + Character N-Gram feature extractor
    word_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 2))
    char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
    
    robust_vectorizer = FeatureUnion([
        ('word', word_vectorizer),
        ('char', char_vectorizer)
    ])
    
    X = robust_vectorizer.fit_transform(df['message'])
    y = df['label_num']

    # Train SVM with balanced weights to penalize spam misclassifications heavily
    svm_model = SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced', random_state=42)
    svm_model.fit(X, y)
    return robust_vectorizer, svm_model

try:
    vectorizer, model = load_robust_classifier_and_vectorizer()
    st.success("Robust Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the classifier: {e}")

# User input
user_input = st.text_area("Paste the email or message content below:", height=150,
                          placeholder="Enter your message here...")

if st.button("Classify Message", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Vectorize input using our robust vectorizer
        transformed_input = vectorizer.transform([user_input])

        # Get probabilities
        probabilities = model.predict_proba(transformed_input)[0]
        spam_probability = probabilities[1]

        # Display Results using a slightly more sensitive threshold (35%) to catch critical financial scams
        st.markdown("--- ")
        if spam_probability >= 0.35:
            st.error(f"🚨 **Result: SPAM** (Confidence: {spam_probability*100:.2f}%)")
        else:
            st.success(f"✅ **Result: HAM** (Confidence: {probabilities[0]*100:.2f}%)")
