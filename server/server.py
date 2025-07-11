from flask import Flask, request, jsonify
import whisper
import requests
import os
import tempfile

app = Flask(__name__)
model = whisper.load_model("base.en")  # small, fast, English-only

OPENROUTER_API_KEY = ""
MODEL_ID = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"

@app.route('/chatgpt-audio', methods=['POST'])
def chatgpt_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Missing audio file"}), 400

    audio_file = request.files['audio']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        audio_path = temp.name
        audio_file.save(audio_path)

    try:
        # Step 1: Transcribe
        result = model.transcribe(audio_path)
        prompt = result['text']
        print(f"Transcribed: {prompt}")

        # Step 2: Send to OpenRouter (Dolphin)
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "esp32-audio-demo"
        }
        body = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}]
        }
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
        r.raise_for_status()
        response_text = r.json()['choices'][0]['message']['content']
        return jsonify({"response": response_text.strip()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(audio_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
