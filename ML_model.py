import tkinter as tk
from tkinter import messagebox, scrolledtext
from tkinter.font import Font
import re
from tkinter import messagebox
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np

# Function to check grammatical errors and suggest corrections
def check_grammar():
    input_text = input_text_box.get("1.0", tk.END).strip()
    
    if not input_text:
        messagebox.showwarning("Input Error", "Please enter a sentence to check.")
        return
    
    # Get the corrected sentence using the grammar correction model
    corrected_sentence = grammar_correction(input_text)
    
    # Display the corrected sentence in the output box
    if corrected_sentence:
        output_text_box.delete("1.0", tk.END)
        output_text_box.insert(tk.END, f"Corrected Sentence: {corrected_sentence}\n")
    else:
        messagebox.showinfo("No Errors", "No grammatical errors found.")

# Sample Grammar Correction Function
def grammar_correction(sentence):
    # Load dictionaries
    def load_dictionary(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return [line.strip() for line in file]
        except FileNotFoundError:
            messagebox.showerror("Error", f"Dictionary file not found at: {file_path}")
            return []
        except ValueError:
            messagebox.showerror("Error", f"Issue with the dictionary file: {file_path}")
            return []

    # Load sentence data
    incorrect = "Grammar-Checker/Incorrect_sentences.txt"
    incorrect_sentences = load_dictionary(incorrect)
    correct = 'Grammar-Checker/Correct_sentences.txt'
    correct_sentences = load_dictionary(correct)

    # Preprocessing function to clean and tokenize sentences
    def preprocess_sentences(sentences):
        sentences = [re.sub(r'[^\w\s]', '', sentence) for sentence in sentences]
        return sentences

    incorrect_sentences = preprocess_sentences(incorrect_sentences)
    correct_sentences = preprocess_sentences(correct_sentences)

    # Create a vocabulary from the sentences
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=False)
    tokenizer.fit_on_texts(incorrect_sentences + correct_sentences)

    # Convert sentences to sequences
    input_sequences = tokenizer.texts_to_sequences(incorrect_sentences)
    output_sequences = tokenizer.texts_to_sequences(correct_sentences)

    # Padding sequences to ensure equal length for batch processing
    max_sequence_length = max([len(seq) for seq in input_sequences + output_sequences])
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
    output_sequences = tf.keras.preprocessing.sequence.pad_sequences(output_sequences, maxlen=max_sequence_length, padding='post')

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(input_sequences, output_sequences, test_size=0.2)

    # Define the Seq2Seq model
    def build_model(input_vocab_size, output_vocab_size, input_seq_len, output_seq_len):
        # Encoder
        encoder_inputs = layers.Input(shape=(input_seq_len,))
        encoder_embedding = layers.Embedding(input_vocab_size, 256)(encoder_inputs)
        encoder_lstm = layers.LSTM(256, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = layers.Input(shape=(output_seq_len,))
        decoder_embedding = layers.Embedding(output_vocab_size, 256)(decoder_inputs)
        decoder_lstm = layers.LSTM(256, return_sequences=True, return_state=True)
        decoder_lstm_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        
        decoder_dense = layers.Dense(output_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_lstm_outputs)
        
        # Model
        model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        return model

    input_vocab_size = len(tokenizer.word_index) + 1
    output_vocab_size = len(tokenizer.word_index) + 1

    model = build_model(input_vocab_size, output_vocab_size, max_sequence_length, max_sequence_length)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Prepare decoder inputs (shifted output sequences for training)
    decoder_inputs_train = np.zeros_like(y_train)
    decoder_inputs_train[:, 1:] = y_train[:, :-1]  # Shift the output sequence

    decoder_inputs_val = np.zeros_like(y_val)
    decoder_inputs_val[:, 1:] = y_val[:, :-1]  # Shift the output sequence

    # Train the model
    model.fit([X_train, decoder_inputs_train], np.expand_dims(y_train, -1), 
            validation_data=([X_val, decoder_inputs_val], np.expand_dims(y_val, -1)),
            batch_size=32, epochs=10)

    # Prediction function (to generate corrected sentences)
    def predict(input_text):
        input_seq = tokenizer.texts_to_sequences([input_text])
        input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=max_sequence_length, padding='post')
        
        # Decoder input is just a sequence of zeros to start the prediction
        decoder_input = np.zeros_like(input_seq)
        
        prediction = model.predict([input_seq, decoder_input])
        predicted_sequence = np.argmax(prediction, axis=-1)[0]
        
        # Convert predicted sequence to sentence
        predicted_sentence = ' '.join([tokenizer.index_word.get(idx, '') for idx in predicted_sequence if idx != 0])
        return predicted_sentence

    # Run the grammar correction model
    corrected_sentence = predict(sentence)
    return corrected_sentence

# Create the main window
window = tk.Tk()
window.title("Tamil Grammar Checker")
window.geometry("600x500")
window.configure(bg="#f0f0f0")

# Set fonts
title_font = Font(family="Helvetica", size=16, weight="bold")
label_font = Font(family="Helvetica", size=12)
button_font = Font(family="Helvetica", size=12, weight="bold")

# Create and place the header label
header_label = tk.Label(window, text="Tamil Grammar Checker", font=title_font, bg="#f0f0f0", fg="#333")
header_label.pack(pady=20)

# Create and place the input label and text box
input_label = tk.Label(window, text="Enter Tamil Sentence:", font=label_font, bg="#f0f0f0", fg="#333")
input_label.pack(pady=10)

input_text_box = scrolledtext.ScrolledText(window, height=6, width=60, wrap=tk.WORD, font=("Arial", 12), bg="#ffffff", fg="#333")
input_text_box.pack(pady=10)

# Create the check button with a modern look
check_button = tk.Button(window, text="Check Grammar", command=check_grammar, font=button_font, bg="#4CAF50", fg="#ffffff", relief="raised", width=20, height=2)
check_button.pack(pady=20)

# Create and place the output label and text box
output_label = tk.Label(window, text="Grammar Errors and Corrections:", font=label_font, bg="#f0f0f0", fg="#333")
output_label.pack(pady=10)

output_text_box = scrolledtext.ScrolledText(window, height=10, width=60, wrap=tk.WORD, font=("Arial", 12), bg="#ffffff", fg="#333")
output_text_box.pack(pady=10)

# Start the Tkinter event loop
window.mainloop()
