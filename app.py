from flask import Flask, render_template, request
from utils.general import Load_model,translate
import torch
import torchtext




app = Flask(__name__)

model_path = "model/best-val-lstm_lm.pt"
params_path = "model/params.pt"
vocab_path ="model/vocab.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Load_model(model_path,params_path).to(device)
model.eval()
vocab = torch.load(vocab_path)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/suggestions')
def generate_suggestions():
    prompt = request.args.get('code', '')
    print(prompt)

    suggestion = translate(prompt)
    return {'suggestions': [f'<li class="list-group-item">{suggestion}</li>']}


if __name__ == '__main__':
    app.run(debug=True)
