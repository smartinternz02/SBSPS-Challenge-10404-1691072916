!pip install tensorflow
!pip install numpy
!pip install scipy
!pip install pandas
!pip install matplotlib
!pip install datasets
!pip install transformers

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the dataset
data = pd.read_csv("BBC_News_processed.csv")

# Preprocess the data
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(data["Text_parsed"])
encoder = LabelEncoder()
y = encoder.fit_transform(data["Category_target"])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the MLP model
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000)
mlp_model.fit(X_train, y_train)

# Evaluate the MLP model
y_pred = mlp_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("MLP Model Accuracy:", accuracy)

!pip install seaborn matplotlib

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Accuracy Distribution
plt.figure(figsize=(8, 6))
sns.histplot(y_pred, color="skyblue", label="Predicted")
sns.histplot(y_test, color="orange", label="Actual")
plt.xlabel("Category")
plt.ylabel("Frequency")
plt.title("Accuracy Distribution")
plt.legend()
plt.show()

# Define a function to predict the category of a given text
def predict_category(text):
    # Preprocess the input text using the same vectorizer
    text_vector = vectorizer.transform([text])

    # Predict the category using the trained MLP model
    category_id = mlp_model.predict(text_vector)[0]

    # Decode the category using the label encoder
    category = encoder.inverse_transform([category_id])[0]

    return category

# Test the function with some sample texts
sample_texts = [
    "This is a sports-related article about soccer.",
    "Science and technology advancements are changing our lives.",
    "The stock market experienced a significant dip yesterday.",
]

for text in sample_texts:
    predicted_category = predict_category(text)
    print(f"Text: {text}\nPredicted Category: {predicted_category}\n")

import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_shuffled_creative_caption_gpt2(text, category):
    prompt = f"Given the text: '{text}' and category: '{category}', generate a unique and creative caption:"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

    generated_caption = tokenizer.decode(output[0], skip_special_tokens=True)

    # Split the generated caption into words and shuffle them
    words = generated_caption.split()
    random.shuffle(words)
    shuffled_caption = ' '.join(words)

    # Limit the caption to 50 words
    caption_words = shuffled_caption.split()[:50]
    final_caption = ' '.join(caption_words)

    return final_caption

text = input('Enter some text: ')
category = input('Enter the category: ')
caption = generate_shuffled_creative_caption_gpt2(text, category)

print('Generated unique, shuffled, and creative caption:')
print(caption)