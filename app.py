from flask import Flask, render_template, request
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

model_name = "google/flan-t5-xl"
local_model_dir = "./flan_t5_model"

# Initialize tokenizer and model globally
tokenizer = None
model = None

# Load or download the model
def load_model():
    global tokenizer, model
    if not os.path.exists(local_model_dir):
        os.makedirs(local_model_dir, exist_ok=True)
        print(f"Downloading model {model_name} for the first time...")
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model.save_pretrained(local_model_dir)
        tokenizer.save_pretrained(local_model_dir)
    else:
        print(f"Loading model {model_name} from local storage...")
        model = T5ForConditionalGeneration.from_pretrained(local_model_dir)
        tokenizer = T5Tokenizer.from_pretrained(local_model_dir)

load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            answer = process_question(question)
            return render_template('index.html', question=question, answer=answer)
    return render_template('index.html', question=None, answer=None)

def process_question(question):
    input_ids = tokenizer(question, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == '__main__':
    app.run(debug=True)
