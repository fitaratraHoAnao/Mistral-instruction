from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import requests

# Charger les variables d'environnement
load_dotenv()

# Configuration de l'application Flask
app = Flask(__name__)
HOST = "0.0.0.0"
PORT = 5000

# Variables d'environnement
API_URL = os.getenv("API_URL", "https://api-inference.huggingface.co/models/mistralai/Mistral-Nemo-Instruct-2407")
API_KEY = os.getenv("API_KEY")

headers = {"Authorization": f"Bearer {API_KEY}"}

@app.route("/mistral", methods=["GET"])
def mistral():
    # Récupérer la question à partir des paramètres de requête
    question = request.args.get("question")
    if not question:
        return jsonify({"error": "Parameter 'question' is required."}), 400

    # Préparer la charge utile pour l'appel API
    payload = {
        "inputs": question,
        "parameters": {"max_new_tokens": 128}
    }

    try:
        # Envoyer la requête POST à l'API
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            # Extraire le texte généré de la réponse
            generated_text = response.json()[0].get("generated_text", "No response text available.")
            return jsonify({"response": generated_text})
        else:
            # Retourner l'erreur de l'API
            return jsonify({
                "error": f"API returned status {response.status_code}",
                "details": response.json()
            }), response.status_code
    except Exception as e:
        # Gérer les exceptions lors de la requête
        return jsonify({
            "error": "An error occurred while processing the request.",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
