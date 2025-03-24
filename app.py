from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os
import re
app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/*": {"origins": "*"}})

HF_API_KEY = "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"
HF_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index = faiss.read_index("aboutMe_faiss_index.bin")

with open("aboutMe.txt", "r", encoding="utf-8") as f:
    sentences = f.read().split(". ")

print("FAISS index loaded with", len(sentences), "entries.")

def search_faiss(query):
    """Find the most relevant sentence using FAISS."""
    query_embedding = model.encode([query]).astype(np.float32)
    _, indices = index.search(query_embedding, 1)
    best_match_idx = indices[0][0]

    if best_match_idx < 0 or best_match_idx >= len(sentences):
        # return None 
        return None, query_embedding

    return sentences[best_match_idx]

def call_huggingface_api(context, query):
    """Use Hugging Face API to refine the answer."""
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": f"Context: {context}\nQuestion: {query}"}

    try:
        response = requests.post(f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}", json=payload, headers=headers)
        response.raise_for_status()
        result=response.json()[0]["generated_text"]
        # return response.json()[0]["generated_text"] 
        match = re.search(r"Answer:\s*(.*)", result, re.IGNORECASE)
        answer = match.group(1).strip() if match else "I'm not sure ds about that."

        return answer
    except Exception as e:
        print(f" Hugging Face API error: {e}")
        return "I'm unable to generate an answer right now."

@app.route("/")
def home():
    """Simple web interface to test API."""
    return render_template("index.html")

@app.route("/test")
def test_api():
    """Hardcoded response for testing API."""
    return jsonify({"message": " Flask API is working!", "sample_response": "This is a test response from the server."})

@app.route("/search", methods=["POST"])
def search():
    """Search FAISS and refine the response using Hugging Face."""
    try:
        data = request.json
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Query is required"}), 400
        if query.lower() == "test":
            return jsonify({"answer": "This is a hardcoded test response!"})

        context = search_faiss(query)
        if not context:
            return jsonify({"error": "No relevant match found"}), 404

        refined_answer = call_huggingface_api(context, query)

        return jsonify({"answer": refined_answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
