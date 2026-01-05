import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="AI Dataset Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------------------------
# Load Models (Cached)
# -------------------------------------------------
@st.cache_resource
def load_models():
    intent_model = joblib.load("models/intent_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    column_embeddings = joblib.load("models/column_embeddings.pkl")
    columns = joblib.load("models/columns.pkl")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    return intent_model, vectorizer, column_embeddings, columns, embed_model


intent_model, vectorizer, column_embeddings, columns, embed_model = load_models()

# -------------------------------------------------
# ML Functions
# -------------------------------------------------
def predict_intent(query):
    q_vec = vectorizer.transform([query])
    return intent_model.predict(q_vec)[0]


def detect_column(query):
    q_emb = embed_model.encode(query)
    scores = np.dot(column_embeddings, q_emb)
    return columns[np.argmax(scores)]

# -------------------------------------------------
# Sidebar – Upload CSV
# -------------------------------------------------
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("⬅️ Upload a CSV file to start chatting with your data")
    st.stop()

df = pd.read_csv(uploaded_file)

# -------------------------------------------------
# Sidebar – Dataset Info
# -------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("📊 Dataset Info")

st.sidebar.metric("Rows", df.shape[0])
st.sidebar.metric("Columns", df.shape[1])

with st.sidebar.expander("📌 Column Names"):
    for col in df.columns:
        st.write(f"- {col}")

st.sidebar.markdown("---")
st.sidebar.header("🧪 Example Queries")
st.sidebar.code("""
average price
highest rating
top 5 highest price
price descending
count records
what's going on in dataset
""")

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🤖 AI Dataset Chatbot</h1>
    <p style="text-align:center; color:gray;">
        ML-powered chatbot to analyze and query datasets
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Session State
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# Render Chat History
# -------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["type"] == "dataframe":
            st.dataframe(msg["content"], use_container_width=True)
        elif msg["type"] == "json":
            st.json(msg["content"])
        else:
            st.write(msg["content"])

# -------------------------------------------------
# Chat Input
# -------------------------------------------------
query = st.chat_input("Ask something about the dataset...")

if query:
    # User message
    st.session_state.messages.append({
        "role": "user",
        "type": "text",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    # ---------------------------------------------
    # ML-based Understanding
    # ---------------------------------------------
    intent = predict_intent(query)
    column = detect_column(query)

    # ---------------------------------------------
    # Action Logic (Lightweight Executor)
    # ---------------------------------------------
    response = None

    if intent == "COUNT":
        response = len(df)

    elif intent == "AVERAGE":
        response = round(df[column].mean(), 2)

    elif intent == "MAX":
        response = df[column].max()

    elif intent == "MIN":
        response = df[column].min()

    elif intent == "SORT_DESC":
        response = df.sort_values(by=column, ascending=False).head(10)

    elif intent == "SORT_ASC":
        response = df.sort_values(by=column, ascending=True).head(10)

    elif intent == "SUMMARY":
        response = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist()
        }

    else:
        response = "🤔 Sorry, I couldn't confidently understand that query."

    # ---------------------------------------------
    # Assistant Response
    # ---------------------------------------------
    intent_msg = f"🧠 **Intent:** `{intent}` | 🎯 **Column:** `{column}`"

    st.session_state.messages.append({
        "role": "assistant",
        "type": "text",
        "content": intent_msg
    })

    with st.chat_message("assistant"):
        st.markdown(intent_msg)

        if isinstance(response, pd.DataFrame):
            st.session_state.messages.append({
                "role": "assistant",
                "type": "dataframe",
                "content": response
            })
            st.dataframe(response, use_container_width=True)

        elif isinstance(response, dict):
            st.session_state.messages.append({
                "role": "assistant",
                "type": "json",
                "content": response
            })
            st.json(response)

        else:
            st.session_state.messages.append({
                "role": "assistant",
                "type": "text",
                "content": response
            })
            st.write(response)
