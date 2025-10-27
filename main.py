import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder

# --- Load dataset ---
dataframes = []
for file in os.listdir("dataset"):
    path = os.path.join("dataset", file)
    if file.endswith(".csv"):
        df = pd.read_csv(path)
    elif file.endswith(".txt"):
        df = pd.read_csv(path, delimiter="\t", header=None, names=["text"])
    else:
        continue
    dataframes.append(df)

dataset = pd.concat(dataframes, ignore_index=True)

# --- Handle datasets with or without labels ---
if "label" in dataset.columns:
    X = dataset["text"].astype(str).values
    y = dataset["label"].astype(str).values
else:
    # If there are no labels, we just generate dummy labels for unsupervised style
    X = dataset["text"].astype(str).values
    y = np.zeros(len(X))

# --- Encode labels ---
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# --- Tokenize text ---
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_seq, maxlen=100, padding="post", truncating="post")

# --- Build model ---
model = keras.Sequential([
    layers.Embedding(10000, 64, input_length=100),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(set(y)), activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# --- Train ---
model.fit(X_padded, y, epochs=5, batch_size=32, validation_split=0.2)

# --- Save model ---
model.save("ai_bot.keras")

# --- Chat function ---
def chat(prompt):
    seq = tokenizer.texts_to_sequences([prompt])
    padded = pad_sequences(seq, maxlen=100, padding="post")
    prediction = model.predict(padded)
    label = encoder.inverse_transform([np.argmax(prediction)])[0]
    return label

# --- Run chat loop ---
while True:
    prompt = input("You: ")
    if prompt.lower() in ["quit", "exit"]:
        break
    response = chat(prompt)
    print("AI:", response)
