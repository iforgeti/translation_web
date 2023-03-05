from flask import Flask, render_template, request
from utils.general import Load_model,translate
import torch
import torchtext
import pickle
from pythainlp import word_tokenize




app = Flask(__name__)

device = torch.device('cpu')
tokenizer = word_tokenize
model = Load_model(device)  
model.eval()

with open("models/vocab_en.pkl",'rb') as f:
    vocab_en = pickle.load(f)

with open("models/vocab_th.pkl",'rb') as f:
    vocab_th = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/suggestions')
def generate_suggestions():
    prompt = request.args.get('code', '')
    print(prompt)

    suggestion = translate(prompt,vocab_th,vocab_en, model,tokenizer)
    return {'suggestions': [f'<li class="list-group-item">{suggestion}</li>']}


if __name__ == '__main__':
    app.run(debug=True)
