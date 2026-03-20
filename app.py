import streamlit as st
from textblob import TextBlob
from datetime import datetime
import pandas as pd
import time

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Sentiment Chatbot Pro", page_icon="🤖", layout="wide")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ Settings")
show_score = st.sidebar.checkbox("Show Sentiment Score", True)

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.session_state.eval_data = []
    st.rerun()

# -------------------------------
# Title
# -------------------------------
st.title("🤖 AI Sentiment Chatbot with Evaluation")
st.write("Now with accuracy tracking & misclassification detection 📊")

# -------------------------------
# Session State
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "eval_data" not in st.session_state:
    st.session_state.eval_data = []

# -------------------------------
# Sentiment Function (Rule-based)
# -------------------------------
def get_sentiment_rule(text):
    text = text.lower()

    if any(w in text for w in ["happy", "good", "great", "awesome"]):
        return "Positive"
    if any(w in text for w in ["sad", "bad", "stress", "tired", "angry"]):
        return "Negative"

    return "Neutral"

# -------------------------------
# TextBlob Sentiment
# -------------------------------
def get_sentiment_blob(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# -------------------------------
# Paragraph Analysis
# -------------------------------
def analyze_paragraph(text):
    sentences = text.split(".")
    results = []
    scores = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            blob = TextBlob(sentence)
            polarity = blob.sentiment.polarity
            scores.append(polarity)

            if polarity > 0:
                sentiment = "Positive 😊"
            elif polarity < 0:
                sentiment = "Negative 😞"
            else:
                sentiment = "Neutral 😐"

            results.append((sentence, sentiment))

    avg_score = sum(scores) / len(scores) if scores else 0

    if avg_score > 0:
        overall = "Overall Positive 😊"
    elif avg_score < 0:
        overall = "Overall Negative 😞"
    else:
        overall = "Overall Neutral 😐"

    return results, overall, avg_score

# -------------------------------
# Chatbot Reply
# -------------------------------
def chatbot_reply(user_input, overall):
    if "hi" in user_input.lower():
        return "Hi! 😊 How are you feeling today?"

    if "Positive" in overall:
        return "That's great! 😄 Keep it up!"

    if "Negative" in overall:
        return "I'm here for you 💙 Stay strong!"

    return "Tell me more 🙂"

# -------------------------------
# Typing Effect
# -------------------------------
def stream_response(text):
    placeholder = st.empty()
    full = ""

    for word in text.split():
        full += word + " "
        placeholder.markdown(full)
        time.sleep(0.03)

    return full

# -------------------------------
# Display Chat
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -------------------------------
# Input
# -------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    time_now = datetime.now().strftime("%H:%M:%S")

    # Analysis
    results, overall, avg_score = analyze_paragraph(user_input)

    rule_pred = get_sentiment_rule(user_input)
    blob_pred = get_sentiment_blob(user_input)

    # Store evaluation
    st.session_state.eval_data.append({
        "text": user_input,
        "rule": rule_pred,
        "blob": blob_pred
    })

    # Reply
    response = chatbot_reply(user_input, overall)

    # Build response
    full_response = "**Sentence-wise Sentiment:**\n"
    for s, senti in results:
        full_response += f"- {s} → {senti}\n"

    full_response += f"\n**Overall Sentiment:** {overall} (Score: {avg_score:.2f})"

    # Misclassification check
    if rule_pred != blob_pred:
        full_response += f"\n\n⚠️ Possible Misclassification Detected!"
        full_response += f"\nRule-Based: {rule_pred} | TextBlob: {blob_pred}"
        full_response += f"\n✔ Suggested Correct Sentiment: {blob_pred}"

    full_response += f"\n\n{response}"

    # Show user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Bot
    with st.chat_message("assistant"):
        st.write("Typing...")
        time.sleep(1)
        final = stream_response(full_response)

    st.session_state.messages.append({"role": "assistant", "content": final})

# -------------------------------
# 📊 ANALYTICS SECTION
# -------------------------------
if st.session_state.eval_data:
    st.subheader("📊 Model Evaluation")

    df = pd.DataFrame(st.session_state.eval_data)

    # Frequency graph
    st.write("### Sentiment Frequency")
    st.bar_chart(df["blob"].value_counts())

    # Accuracy calculation
    correct = (df["rule"] == df["blob"]).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0

    st.write(f"### Accuracy: {accuracy*100:.2f}%")

    # Misclassified
    misclassified = df[df["rule"] != df["blob"]]

    if not misclassified.empty:
        st.write("### ⚠️ Misclassified Inputs")
        st.dataframe(misclassified)

# -------------------------------
# Download
# -------------------------------
if st.session_state.messages:
    chat_text = ""
    for msg in st.session_state.messages:
        chat_text += f"{msg['role'].upper()}: {msg['content']}\n"

    st.download_button("📥 Download Chat", chat_text)