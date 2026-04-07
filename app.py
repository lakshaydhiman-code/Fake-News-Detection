import joblib
from flask import Flask, render_template, request
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- Configuration ---
MODEL_FILE = 'final_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'

# --- Model Loading ---
try:
    # Load the saved model and vectorizer
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    print("Model and Vectorizer loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model or Vectorizer file not found. Please run model_training.py first.")
    exit()

# --- Text Preprocessing Function (must match the training script!) ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# --- Flask Application Setup ---
app = Flask(__name__)

# Route for the homepage (input form)
@app.route('/')
def home():
    return render_template('index.html', result=None)

# Route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Get user input
        news_text = request.form['news_text']

        # 2. Preprocess the input
        cleaned_text = preprocess_text(news_text)

        # 3. Vectorize the input
        # Note: We use the loaded 'vectorizer' to TRANSFORM, not FIT_TRANSFORM
        vectorized_text = vectorizer.transform([cleaned_text])

        # 4. Make the prediction
        prediction = model.predict(vectorized_text)[0]
        
        # 5. Determine the human-readable result
        if prediction == 1:
            result = "REAL NEWS"
            style = "text-success"
        else:
            result = "FAKE NEWS"
            style = "text-danger"

        # 6. Render the template with the result
        return render_template('index.html', result=result, style=style, news_text=news_text)

if __name__ == '__main__':
    # Run the Flask app on default host/port
    app.run(debug=True)

# End of app.py