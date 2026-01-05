import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

intent_model = joblib.load("models/intent_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

column_embeddings = joblib.load("models/column_embeddings.pkl")
columns = joblib.load("models/columns.pkl")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def predict_intent(query):
    q_vec = vectorizer.transform([query])
    return intent_model.predict(q_vec)[0]

def detect_column(query):
    q_emb = embed_model.encode(query)
    scores = np.dot(column_embeddings, q_emb)
    return columns[np.argmax(scores)]
