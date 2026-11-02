# ✉️ Spam Email Classifier (Streamlit App)

An interactive Machine Learning web application designed to detect and filter out spam emails/messages based on their textual content. The model utilizes advanced NLP techniques like TF-IDF Vectorization and compares **Multinomial Naive Bayes** and **Support Vector Machine (SVM)** algorithms for robust classification.

## 🚀 Project Overview
* **Task:** Binary classification (Spam vs. Ham)
* **Dataset:** 5,500+ SMS/Email records labeled as Spam or Ham.
* **Key Models:** 
  * **Support Vector Machine (SVM):** ~98.39% accuracy
  * **Multinomial Naive Bayes:** ~97.04% accuracy
* **Interface:** Streamlit Web UI for real-time validation.

## 🛠️ Installation & Setup
To run this project locally, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/Darkstrike03/SPAM.git
cd SPAM
```

### 2. Install dependencies
Create a virtual environment (optional but recommended) and run:
```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
```

### 3. Run the Streamlit Web Application
```bash
streamlit run app.py
```

## 📂 File Directory
* `app.py` - Core Streamlit web application source code.
* `Spam_Email_Classifier.ipynb` - Jupyter notebook containing data exploration, model training, and performance visualization.
* `README.md` - Documentation of the project.

## 📊 Results Summary
| Model | Accuracy | Spam F1-Score | Special Attributes |
| :--- | :---: | :---: | :--- |
| **SVM** | **98.39%** | **0.94** | Higher overall sensitivity to catch spam. |
| **Naive Bayes** | **97.04%** | **0.88** | Perfect precision (0 false spam flags). |
