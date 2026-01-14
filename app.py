from flask import Flask, render_template, request
import pickle
import re
import nltk
import os
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    message = ""

    if request.method == "POST":
        message = request.form["message"]
        cleaned = clean_text(message)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]
        prediction = "Spam 🚫" if result == 1 else "Not Spam ✅"

    return render_template("index.html", prediction=prediction, message=message)

if __name__ == "__main__":
    app.run(debug=True)
