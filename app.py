from flask import Flask, jsonify, request
from flask_cors import CORS
import traceback
from web_qa import answer_questions

app = Flask(__name__)
# Enable CORS for all domains on all routes
CORS(app)
# Or, to enable CORS only for http://localhost:3000
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route('/ask', methods=['POST'])
def ask():
    try:
        # Parse the incoming JSON data
        data = request.json
        question = data.get('question', '')
        context = data.get('context', '')  # Retrieve context passed from the frontend

        # Pass both the question and context to the function that handles answer generation
        response = answer_questions(question, context)
        return jsonify({'answer': response})
    except Exception as e:
        # Use traceback to print the exception to the console
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

    