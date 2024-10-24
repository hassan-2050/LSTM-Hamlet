import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import os

# -------------------------------
# 1. Page Configuration and Styling
# -------------------------------

st.set_page_config(
    page_title="🔮 Hamlet's Next Word Predictor",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Add custom CSS for additional styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        color: white;
        background-color: #2e86de;
    }
    .stTextInput>div>div>input {
        color: #333333;
        background-color: #ffffff;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: #888888;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# 2. Caching Models and Tokenizer
# -------------------------------

@st.cache_resource
def load_lstm_model():
    return load_model('next_word_lstm.h5')

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

model = load_lstm_model()
tokenizer = load_tokenizer()

# -------------------------------
# 3. Next Word Prediction Function
# -------------------------------

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Unknown"

# -------------------------------
# 4. Sidebar - About and Fixed Image Display
# -------------------------------

st.sidebar.header("🎭 About the App")
st.sidebar.info(
    """
    This application uses a trained LSTM model to predict the next word in a given sequence from Shakespeare's *Hamlet*.
    
    **Developed by:** Hassan Shah  
    **Contact:** syedhassan2050@gmail.com 
    """
)

# Fixed Image Display Section
st.sidebar.header("📷 Hamlet Visual")
default_image_path = 'hamlet.jpg'  # Ensure this image exists in your app directory

if os.path.exists(default_image_path):
    try:
        image = Image.open(default_image_path)
        st.sidebar.image(image, caption="Hamlet", use_column_width=True)
    except Exception as e:
        st.sidebar.error(f"Error loading image: {e}")
else:
    st.sidebar.warning("Default image not found. Please ensure 'hamlet_default.jpg' is in the app directory.")

# -------------------------------
# 5. Main App Layout
# -------------------------------

# Title with Emoji
st.title("🔮 Hamlet's Next Word Predictor")

# Description
st.markdown("""
Predict the next word in a sequence from Shakespeare's *Hamlet* using a trained LSTM model.
""")

# Input Section
default_input = "To be, or not to be, that is the"
input_text = st.text_input(
    "Enter a sequence of words from Hamlet",
    default_input,
    help="Type a sequence of words from Hamlet, and the model will predict the next word."
)

# Prediction Button with Icon
if st.button("✨ Predict Next Word"):
    if input_text.strip() == "":
        st.error("❌ Please enter a valid sequence of words.")
    else:
        with st.spinner('🔍 Predicting...'):
            try:
                max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
                next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
                if next_word:
                    st.success(f'**Next word:** `{next_word}`')
                else:
                    st.warning("⚠️ Unable to predict the next word.")
            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")

# Optional: Display Multiple Predictions (Top 3)
def predict_top_n_words(model, tokenizer, text, max_sequence_len, n=3):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)[0]
    top_indices = predicted.argsort()[-n:][::-1]
    index_word = {index: word for word, index in tokenizer.word_index.items()}
    top_words = [(index_word.get(i, "Unknown"), predicted[i]) for i in top_indices]
    return top_words

# Uncomment the following block to enable top N predictions
# if st.button("🔮 Predict Top 3 Next Words"):
#     if input_text.strip() == "":
#         st.error("❌ Please enter a valid sequence of words.")
#     else:
#         with st.spinner('🔍 Predicting...'):
#             try:
#                 max_sequence_len = model.input_shape[1] + 1
#                 top_words = predict_top_n_words(model, tokenizer, input_text, max_sequence_len, n=3)
#                 if top_words:
#                     st.success("**Top 3 Next Words:**")
#                     for word, prob in top_words:
#                         st.write(f"- `{word}` (Probability: {prob:.4f})")
#                 else:
#                     st.warning("⚠️ Unable to predict the next words.")
#             except Exception as e:
#                 st.error(f"⚠️ An error occurred: {e}")

# -------------------------------
# 6. Footer
# -------------------------------

st.markdown(
    """
    ---
    <div class="footer">
        © 2024 Hamlet's Next Word Predictor App
    </div>
    """,
    unsafe_allow_html=True
)
