import tkinter as tk
from tkinter import ttk, messagebox
import Levenshtein

# Load Tamil dictionary
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

# Suggest corrections using Levenshtein Distance
def suggest_correction(word, dictionary):
    suggestions = sorted(dictionary, key=lambda x: Levenshtein.distance(word, x))
    return suggestions[:5]  # Top 5 suggestions

# Check spelling and display suggestions
def check_spelling():
    input_text = text_entry.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showinfo("Info", "Please enter some Tamil text to check.")
        return

    words = input_text.split()
    incorrect_words = []
    suggestions = {}

    for word in words:
        if word not in tamil_words:
            incorrect_words.append(word)
            suggestions[word] = suggest_correction(word, tamil_words)
    
    # Display results
    if incorrect_words:
        result_text = "Spelling Errors Found:\n\n"
        for word in incorrect_words:
            result_text += f"Word: {word}\nSuggestions: {', '.join(suggestions[word])}\n\n"
    else:
        result_text = "All words are correct!"

    result_display.config(state="normal")
    result_display.delete("1.0", tk.END)
    result_display.insert(tk.END, result_text)
    result_display.config(state="disabled")

# GUI Setup
root = tk.Tk()
root.title("Tamil Spell Checker")
root.geometry("800x600")
root.configure(bg="#e6f7ff")

# Load dictionary
file_path = r"Tamil-Spell-and-Grammar-Checker\Dictionary.txt"
tamil_words = load_dictionary(file_path)

# Header
header = tk.Label(root, text="Tamil Spell Checker", font=("Helvetica", 24, "bold"), bg="#e6f7ff", fg="#007acc")
header.pack(pady=20)

# Input Text Area
text_frame = tk.Frame(root, bg="#e6f7ff")
text_frame.pack(pady=10)

text_label = tk.Label(text_frame, text="Enter Tamil Text:", font=("Helvetica", 14), bg="#e6f7ff", fg="#005580")
text_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)

text_entry = tk.Text(text_frame, height=10, width=70, font=("Helvetica", 14))
text_entry.grid(row=1, column=0, padx=10, pady=5)

# Check Button
check_button = ttk.Button(root, text="Check Spelling", command=check_spelling)
check_button.pack(pady=30)

# Output Result Display
result_frame = tk.Frame(root, bg="#e6f7ff")
result_frame.pack(pady=10, fill="both", expand=True)

result_label = tk.Label(result_frame, text="Results:", font=("Helvetica", 14), bg="#e6f7ff", fg="#005580")
result_label.pack(anchor="w", padx=10, pady=5)

result_display = tk.Text(result_frame, height=15, width=70, font=("Helvetica", 14), state="disabled", bg="#f0f4f8")
result_display.pack(padx=10, pady=5)

# Scrollbar for Result Display
scrollbar = ttk.Scrollbar(result_frame, command=result_display.yview)
scrollbar.pack(side="right", fill="y")
result_display["yscrollcommand"] = scrollbar.set

# Start the GUI
root.mainloop()
