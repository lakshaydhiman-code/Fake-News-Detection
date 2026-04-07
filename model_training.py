import pandas as pd
import numpy as np
import swifter
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib
import os 

# --- Configuration ---
TRUE_NEWS_PATH = 'true.csv'
FAKE_NEWS_PATH = 'fake.csv'
MODEL_FILE = 'final_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'

# --- NLTK Check ---
try:
    stopwords.words('english')
    print("NLTK Stopwords check passed.")
except LookupError:
    print("NLTK Stopwords not found. Attempting download...")
    nltk.download('stopwords')
    
# Text Cleaning Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# --- Data Loading and Preparation with Error Checking ---
print("1. Loading Data...")

# Check if files exist before trying to load them
if not os.path.exists(TRUE_NEWS_PATH):
    print(f"ERROR: The file '{TRUE_NEWS_PATH}' was not found in the current directory.")
    print("Please make sure you have downloaded the required CSV files.")
    exit()

if not os.path.exists(FAKE_NEWS_PATH):
    print(f"ERROR: The file '{FAKE_NEWS_PATH}' was not found in the current directory.")
    print("Please make sure you have downloaded the required CSV files.")
    exit()

try:
    df_true = pd.read_csv(TRUE_NEWS_PATH)
    df_fake = pd.read_csv(FAKE_NEWS_PATH)
    
    # Assign labels and combine
    df_true['label'] = 1  
    df_fake['label'] = 0  
    df = pd.concat([df_fake, df_true], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)
    df.dropna(subset=['title', 'text'], inplace=True) # Important: drop NaNs
    df['content'] = df['title'] + ' ' + df['text']
    
    print(f"   -> Data loaded successfully. Total rows: {len(df)}")
    
except Exception as e:
    print(f"A major error occurred during data loading or combination: {e}")
    exit()


# Apply preprocessing
print("2. Preprocessing text...")
try:
   
    df['content'] = df['content'].swifter.apply(preprocess_text)
    print(f"   -> Preprocessing complete. Ready for ML steps.")
except Exception as e:
    print(f"An error occurred during text preprocessing: {e}")
    exit()

# --- Feature Engineering and Splitting ---
print("3. Splitting data and initializing TF-IDF...")
X = df['content']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
print("4. Fitting TfidfVectorizer... (This may take a moment)")
tfidf_vectorizer = TfidfVectorizer(max_features=2500)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
print(f"   -> TF-IDF fitted. Total features: {tfidf_train.shape[1]}")

# --- Model Training and Evaluation ---
print("5. Training Passive Aggressive Classifier... (Wait time depends on CPU)")
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f"-> Model Training COMPLETE. Accuracy: {round(score*100, 2)}%") 

# --- Serialization (Saving Artifacts) ---
print("6. Saving Model and Vectorizer...")
joblib.dump(pac, MODEL_FILE)
joblib.dump(tfidf_vectorizer, VECTORIZER_FILE)
print(f"-> Model saved as {MODEL_FILE}")
print(f"-> Vectorizer saved as {VECTORIZER_FILE}")

# End of model_training.py