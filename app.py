from flask import Flask, request, jsonify
import joblib
import re
import os
from langdetect import detect
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# Inisialisasi Flask app
app = Flask(__name__)

# Load vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load stopwords dan stemmer
import nltk
nltk.download('stopwords')

stopwords_id = set(stopwords.words('indonesian'))
stopwords_en = set(stopwords.words('english'))
stop_words = stopwords_id.union(stopwords_en)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    lang = detect(text)
    if lang == "id":
        tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

def extract_top_keywords(vector, feature_names, top_n=5):
    indices = vector.toarray().argsort()[0][-top_n:][::-1]
    return [feature_names[i] for i in indices]

@app.route('/keywords', methods=['POST'])
def get_keywords():
    data = request.get_json()
    text = data.get("text", "")
    cleaned = clean_text(text)
    tfidf_vector = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    keywords = extract_top_keywords(tfidf_vector, feature_names)
    return jsonify({"keywords": keywords})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
