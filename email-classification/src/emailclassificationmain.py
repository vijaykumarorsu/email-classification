import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import tkinter as tk
from tkinter import messagebox

# Load and preprocess data
def load_data():
    data = pd.read_csv('dataset/emails_sample.csv')
    X = data['text']
    y = data['category']
    return X, y

# Train the model
def train_model(X_train, y_train):
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    return model

# Predict the category of the email
def predict():
    text = text_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Error", "Please enter an email message.")
        return

    prediction = model.predict([text])[0]
    result_label.config(text=f"Predicted Category: {prediction}")

# Create the GUI
def create_gui():
    global model, text_entry, result_label
    
    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # GUI Setup
    window = tk.Tk()
    window.title("Email Classification")
    
    tk.Label(window, text="Enter Email Message:").pack(pady=10)
    
    text_entry = tk.Text(window, height=10, width=50)
    text_entry.pack(pady=10)
    
    predict_button = tk.Button(window, text="Predict Category", command=predict)
    predict_button.pack(pady=10)
    
    result_label = tk.Label(window, text="Predicted Category: ")
    result_label.pack(pady=10)
    
    window.mainloop()

if __name__ == "__main__":
    create_gui()
