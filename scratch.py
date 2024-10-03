import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import random

# Step 2: Prepare the Training Data
with open("C:\\Users\\ABHINAV\\Documents\\story_data.txt", 'r') as file:
    stories = file.read().splitlines()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(stories)

total_words = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(stories)

input_sequences = []
for sequence in sequences:
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max(len(seq) for seq in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Step 6: Build the LSTM Model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Step 7: Train the Model
model.fit(X, y, epochs=20, verbose=1)

# Step 8: Generate Stories
def generate_story(seed_text, num_words, temperature=1.0):
    generated_text = seed_text
    for _ in range(num_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=max_sequence_len - 1, padding='pre')
        predictions = model.predict(encoded)[0]

        if temperature > 0:
            predictions = np.power(predictions, 1.0 / temperature)
            predictions /= np.sum(predictions)

        predicted_word_index = np.random.choice(len(predictions), p=predictions)

        if predicted_word_index == 0:  # Special case for the start token
            predicted_word_index = np.argmax(predictions[1:]) + 1

        output_word = tokenizer.index_word[predicted_word_index]

        # Check if the predicted word is a punctuation mark
        if output_word in ['.', '!', '?']:
            generated_text += output_word + ' '
            break  # End the generation at a punctuation mark
        elif output_word in [',', ';', ':']:
            generated_text = generated_text.rstrip() + output_word + ' '
        else:
            generated_text += ' ' + output_word

        seed_text += ' ' + output_word

    return generated_text

seed = "Once upon a time"
generated_story = generate_story(seed, num_words=200, temperature=0.8)
print(generated_story)