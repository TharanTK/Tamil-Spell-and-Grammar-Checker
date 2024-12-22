import tkinter as tk
from tkinter import messagebox, scrolledtext
from tkinter.font import Font

# Function to check grammatical errors and suggest corrections
def check_grammar():
    input_text = input_text_box.get("1.0", tk.END).strip()
    
    if not input_text:
        messagebox.showwarning("Input Error", "Please enter a sentence to check.")
        return
    
    # Replace this with your grammar-checking logic
    corrections = grammar_correction(input_text)
    
    # Display the results in the output box
    if corrections:
        output_text_box.delete("1.0", tk.END)
        for error, correction in corrections:
            output_text_box.insert(tk.END, f"Error: {error}\nCorrection: {correction}\n\n")
    else:
        messagebox.showinfo("No Errors", "No grammatical errors found.")

# Sample Grammar Correction Function
def grammar_correction(sentence):
    import re

import re

# Function to check for singular/plural agreement (Expanded)
def check_singular_plural(sentence):
    corrections = []
    
    # Define common plural subjects and verbs
    plural_subjects = ["நாங்கள்", "அவர்கள்", "உங்கள்"]
    singular_subjects = ["நான்", "அவன்", "அவள்", "என்"]
    plural_verbs = ["வருகிறோம்", "போகிறோம்", "படிக்கிறோம்", "சொல்கிறோம்"]
    singular_verbs = ["வருகிறேன்", "போகிறேன்", "படிக்கிறேன்", "சொல்கிறேன்"]
    
    # Singular subject with plural verb
    for subject in singular_subjects:
        if subject in sentence:
            if any(verb in sentence for verb in plural_verbs):
                corrections.append((f"{subject} வருகிறோம்", f"{subject} வருகிறேன்"))

    # Plural subject with singular verb
    for subject in plural_subjects:
        if subject in sentence:
            if any(verb in sentence for verb in singular_verbs):
                corrections.append((f"{subject} வருகிறேன்", f"{subject} வருகிறோம்"))
    
    # Return found corrections
    return corrections


# Function to check for tense errors (Expanded)
def check_tense_errors(sentence):
    corrections = []
    
    # Tense errors for present vs past vs future
    present_tense_verbs = ["வருகிறேன்", "போகிறேன்", "படிக்கிறேன்", "சொல்கிறேன்"]
    past_tense_verbs = ["வந்தேன்", "போனேன்", "படித்தேன்", "சொன்னேன்"]
    future_tense_verbs = ["வரப்போகிறேன்", "போகப்போகிறேன்", "படிக்கப்போகிறேன்"]
    
    # Detect present tense in future/past context
    if any(verb in sentence for verb in present_tense_verbs) and "நேற்று" in sentence:
        corrections.append(("வருகிறேன்", "வந்தேன்"))  # Suggest changing to past tense
    
    # Detect past tense in present/future context
    if any(verb in sentence for verb in past_tense_verbs) and "இப்போது" in sentence:
        corrections.append(("வந்தேன்", "வருகிறேன்"))  # Suggest changing to present tense
    
    # Detect future tense in past context
    if any(verb in sentence for verb in future_tense_verbs) and "நேற்று" in sentence:
        corrections.append(("வரப்போகிறேன்", "வந்தேன்"))  # Suggest changing to past tense
    
    return corrections


# Function to check word order errors
def check_word_order(sentence):
    corrections = []
    
    # Check for verb misplacement (Verb should generally be at the end in Tamil sentences)
    if re.search(r"(\w+)\s(வருகிறேன்|போகிறேன்|படிக்கிறேன்|சொல்கிறேன்)", sentence):
        corrections.append(("Verb misplacement detected", "Consider moving the verb to the end."))
    
    return corrections


# Function to handle more complex sentence structure issues
def check_sentence_structure(sentence):
    corrections = []
    
    # Check for missing subject
    if "வருகிறேன்" in sentence or "போகிறேன்" in sentence:
        if not any(subject in sentence for subject in ["நான்", "நாங்கள்", "அவன்", "அவள்"]):
            corrections.append(("Missing subject", "Add a subject like 'நான்' or 'நாங்கள்'"))
    
    # Check for improper negation (e.g., incorrect negation like "நான் வரவில்லை")
    if re.search(r"வரவில்லை", sentence) and "நான்" not in sentence:
        corrections.append(("Incorrect negation", "Recheck the usage of negation."))
    
    return corrections


# Main grammar correction function
def grammar_correction(sentence):
    input_text = input_text_box.get("1.0", tk.END).strip()
    corrections = []
    
    # Apply singular/plural agreement check
    corrections.extend(check_singular_plural(sentence))
    
    # Apply tense error check
    corrections.extend(check_tense_errors(sentence))
    
    # Check for word order errors
    corrections.extend(check_word_order(sentence))
    
    # Check for structural issues in the sentence
    corrections.extend(check_sentence_structure(sentence))
    
    return corrections


    for sentence in input_text:
     print(f"Original Sentence: {sentence}")
     corrections = grammar_correction(sentence)
    
     if corrections:
            for error, correction in corrections:
                print(f"Error: {error}\nCorrection: {correction}\n")
            else:
                 print("No grammatical errors found.\n")



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
