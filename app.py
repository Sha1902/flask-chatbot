from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from transformers import pipeline
from data_processor import process_data
import os

app = Flask(__name__)
api = Api(app)

# Load vector store (if still needed)
vectorstore = process_data()

# Load local GPT-2 model
llm = pipeline("text-generation", model="gpt2")

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Chatbot API. Use POST /chat to interact."})

class Chatbot(Resource):
    def post(self):
        data = request.get_json()
        question = data.get('question')

        if not question:
            return {"error": "Please provide a question."}, 400

        try:
            response = llm(question, max_length=100, do_sample=True)
            result = response[0]["generated_text"]
            return {"answer": result}, 200
        except Exception as e:
            return {"error": str(e)}, 500

api.add_resource(Chatbot, '/chat')

if __name__ == '__main__':
    app.run(debug=True)
