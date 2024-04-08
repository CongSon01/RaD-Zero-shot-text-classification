from flask import Flask, render_template, request
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
from datasets import Dataset


app = Flask(__name__)

# Load the saved model
model_path = "../bert-finetuned-sem_eval/checkpoint-2780/"
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        return render_template('result.html', user_input=predict(user_input))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)