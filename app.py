from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os

app = Flask(__name__)
CORS(app)

# --- API KEY FROM ENVIRONMENT VARIABLE ---
# We store it in an environment variable for security
HARDCODED_KEY = os.environ.get('OPENAI_API_KEY', None)

@app.route('/', methods=['GET'])
def home():
    return "<h1>Server is Online</h1><p>API Key is hardcoded.</p>"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        messages = data.get('messages', [])
        
        # GET THE KEY FROM ENVIRONMENT VARIABLE
        api_key = HARDCODED_KEY
        if not api_key:
            print("Error: No API key found in environment variable OPENAI_API_KEY.")
            return jsonify({'error': 'No API key found. Set OPENAI_API_KEY environment variable.'}), 401
        
        print(f"Using hardcoded key starting with: {api_key[:15]}...")
        print(f"Sending {len(messages)} messages to OpenAI...")

        # Create client
        client = openai.OpenAI(api_key=api_key)
        
        # Send to OpenAI
        response = client.chat.completions.create(
            model='gpt-4o-mini', # Make sure you have credit for this model
            messages=messages,
            max_tokens=500
        )
        
        reply = response.choices[0].message.content
        print("Success! Reply received.")
        
        return jsonify({'success': True, 'reply': reply})
        
    except openai.AuthenticationError:
        print("Error: 401 Authentication Failed. The key itself is invalid or expired.")
        return jsonify({'error': 'Authentication failed. Check the hardcoded key.'}), 401
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting server on 127.0.0.1...")
    app.run(debug=True, port=5000, host='127.0.0.1')