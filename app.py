import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# -----------------------------
# 🧠 Load Model and Tokenizer
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('next_word_lstm.h5')
    return model

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()

# -----------------------------
# 🔮 Prediction Function
# -----------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    if not text.strip():
        return "Please enter some text."
    
    # Convert input text to sequence
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Trim if input is longer than max_sequence_len
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    # Pad sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    # Predict next word probabilities
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    # Find word corresponding to predicted index
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return "Not found"

# -----------------------------
# 🎨 Streamlit UI
# -----------------------------
st.set_page_config(page_title="Next Word Predictor", page_icon="🧠", layout="centered")

st.title("🧠 LSTM Next Word Predictor")
st.write("Enter a few words and let the model predict the next one!")

user_input = st.text_input("✍️ Type your text here:")

if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Assuming model was trained on sequences of fixed length
        max_seq_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, user_input, max_seq_len)
        st.success(f"**Predicted next word:** {next_word}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & TensorFlow")
