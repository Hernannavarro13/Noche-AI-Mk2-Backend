from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from frontend

# Load model
tokenizer = AutoTokenizer.from_pretrained("JetBrains/Mellum-4b-base")
model = AutoModelForCausalLM.from_pretrained("JetBrains/Mellum-4b-base")
model.eval()

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
        
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": result})
