# 🧠 LSTM Next Word Predictor (Hamlet Text)

A **Next Word Prediction** model built using **LSTM (Long Short-Term Memory)** — trained on text from Shakespeare’s *Hamlet*.  
The model learns how words follow each other and predicts the most likely next word for a given input sequence.

---

## 📘 Project Overview

This project demonstrates how an **LSTM neural network** can learn language patterns from text and generate new text by predicting the next word in a sequence.  

We train the model on the **Hamlet** dataset — one of Shakespeare’s most famous works — to teach it the style and structure of Shakespearean English.

---

## 🚀 Features

- Tokenizes and preprocesses raw Hamlet text  
- Generates **n-gram sequences** for next-word prediction  
- Trains an **LSTM neural network** on tokenized sequences  
- Predicts the **next word** for any given phrase  
- Supports **auto text generation** (one word at a time)

---

## 🧩 How It Works

1. **Data Preprocessing**
   - Load and clean the Hamlet text file
   - Convert words to numeric tokens using Keras `Tokenizer`
   - Create *n-gram sequences* so the model can learn context

2. **Training Data Preparation**
   - Each sequence’s last word is the **label (y)**
   - All preceding words form the **predictor (x)**
   - Example:
     | Input (x) | Label (y) |
     |------------|------------|
     | I am | pro |
     | I am pro | coder |

3. **Model Architecture**
   - Embedding Layer (for word representation)
   - LSTM Layer (for sequence learning)
   - Dense Output Layer (softmax activation for next-word prediction)

4. **Prediction**
   - Feed a partial sentence like `"To be or not"`
   - Model predicts `"to"` as the next word

---

## 🧠 Example Workflow

```python
# Sample prediction
seed_text = "To be or not"
next_word = predict_next_word(model, tokenizer, seed_text)
print(f"{seed_text} {next_word}")
```

**Output:**
```
To be or not to
```

---

## 📦 Requirements

Make sure you have the following installed:

```bash
pip install tensorflow numpy keras
```

---

## 🧰 Code Structure

```
📂 LSTM_Next_Word_Predictor/
│
├── 📜 hamlet.txt               # Training dataset
├── 🧠 lstm_next_word.py        # Model training and prediction code
├── 📘 README.md                # Documentation
└── 📊 saved_model.h5           # Trained model (optional)
```

---

## ⚙️ Key Code Snippets

### Creating n-gram Sequences
```python
input_sequences = []
for line in data.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
```

### Splitting Predictors and Labels
```python
x = pad_inp_sequence[:, :-1]
y = pad_inp_sequence[:, -1]
```

### Defining the LSTM Model
```python
model = Sequential([
    Embedding(total_words, 64, input_length=max_seq_len-1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])
```

### Predicting the Next Word
```python
def predict_next_word(model, tokenizer, text, max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)[0]
    return tokenizer.index_word[predicted]
```

---

## 📈 Training Details

- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Activation:** Softmax  
- **Epochs:** 100–500 (depending on dataset size)  

---

## 🗣️ Example Outputs

| Input | Predicted Next Word |
|--------|----------------------|
| To be or not | to |
| The king hath | slain |
| What a piece of | work |

---

## 🧾 Notes

- The quality of predictions improves with more epochs.  
- Larger context windows (more tokens per input) help with grammar consistency.  
- The model can be fine-tuned on modern English texts for different styles.

---

## 🧑‍💻 Author

**Mohit Bohra**  
Built with ❤️ using TensorFlow and Shakespeare’s words.

---

## 📜 License

This project is open-source and available under the **MIT License**.
