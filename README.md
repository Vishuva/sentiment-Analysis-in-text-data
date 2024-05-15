import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from transformers import BertTokenizer, TFBertModel, BertConfig

# Load dataset (e.g., IMDB movie reviews)
# Replace this with your dataset loading code
# Assuming the dataset has 'review' and 'sentiment' columns
df = pd.read_csv('movie_reviews.csv')

# Tokenize and pad sequences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode text data
encoded_data = tokenizer.batch_encode_plus(
    df['review'].tolist(),
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='tf'
)

X = {
    'input_ids': encoded_data['input_ids'],
    'attention_mask': encoded_data['attention_mask']
}

# Convert sentiment labels to numeric
y = np.array(df['sentiment'].map({'positive': 1, 'negative': 0}))

# Define LSTM model
lstm_model = Sequential([
    Embedding(input_dim=len(tokenizer.vocab), output_dim=128, input_length=128),
    Bidirectional(LSTM(units=64, return_sequences=True)),
    Bidirectional(LSTM(units=64)),
    Dense(units=1, activation='sigmoid')
])

# Compile model
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
lstm_model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

# Load pre-trained BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Define input layers
input_ids = tf.keras.layers.Input(shape=(128,), name='input_ids', dtype='int32')
attention_mask = tf.keras.layers.Input(shape=(128,), name='attention_mask', dtype='int32')

# Define BERT layer
output = bert_model([input_ids, attention_mask])[1]

# Define output layer
output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

# Define model
bert_finetuned_model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

# Compile model
bert_finetuned_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
bert_finetuned_model.fit(X, y, epochs=3, batch_size=32, validation_split=0.2)
