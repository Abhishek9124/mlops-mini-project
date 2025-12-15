from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===============================
# NLTK Setup (IMPORTANT)
# ===============================
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ===============================
# Text Preprocessing
# ===============================

def lower_case(text):
    return text.lower()


def removing_urls(text):
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def removing_numbers(text):
    return "".join([char for char in text if not char.isdigit()])


def removing_punctuations(text):
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub("\s+", " ", text).strip()
    return text


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def normalize_text(text):
    text = lower_case(text)
    text = removing_urls(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = remove_stop_words(text)
    text = lemmatization(text)
    return text


# ===============================
# MLflow + DagsHub Configuration
# ===============================

DAGSHUB_TOKEN = os.getenv("DAGSHUB_PAT")
if not DAGSHUB_TOKEN:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_TOKEN
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

DAGSHUB_URL = "https://dagshub.com"
REPO_OWNER = "Abhishek9124"
REPO_NAME = "mlops-mini-project"

mlflow.set_tracking_uri(
    f"{DAGSHUB_URL}/{REPO_OWNER}/{REPO_NAME}.mlflow"
)

# ===============================
# Load Model (Alias-Based )
# ===============================

MODEL_NAME = "mlops_model"
MODEL_ALIAS = "prod"

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

try:
    model = mlflow.pyfunc.load_model("models:/mlops_model/Staging")
except Exception as e:
    raise RuntimeError(
        f"Failed to load model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'. "
        f"Ensure alias is set in MLflow registry.\n{e}"
    )

# ===============================
# Load Vectorizer
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "models", "vectorizer.pkl")

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"vectorizer.pkl not found at {VECTORIZER_PATH}")

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


# ===============================
# Flask App
# ===============================

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")

    if not text.strip():
        return render_template("index.html", result="Invalid input")

    # Preprocess
    clean_text = normalize_text(text)

    # Vectorize
    features = vectorizer.transform([clean_text])
    features_df = pd.DataFrame(
        features.toarray(),
        columns=[str(i) for i in range(features.shape[1])]
    )

    # Predict
    prediction = model.predict(features_df)

    return render_template("index.html", result=prediction[0])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
