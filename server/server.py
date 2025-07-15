from flask import Flask, request, send_file, jsonify
import requests
import os
import io
import traceback

app = Flask(__name__)

OPENAI_API_KEY = ""  # your actual key here
TRANSCRIBE_MODEL = "whisper-1"
CHAT_MODEL = "gpt-4o"
TTS_MODEL = "tts-1"  # or "tts-1-hd"
TTS_VOICE = "nova"   # options: alloy, echo, fable, onyx, nova, shimmer

@app.route('/chatgpt-audio', methods=['POST'])
def chatgpt_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Missing audio file"}), 400

    audio_file = request.files['audio']

    try:
        # Step 1: Transcribe
        transcribe_headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        transcribe_files = {
            "file": (audio_file.filename, audio_file.stream, audio_file.mimetype),
        }
        transcribe_data = {
            "model": TRANSCRIBE_MODEL,
            "response_format": "json"
        }

        transcribe_response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=transcribe_headers,
            files=transcribe_files,
            data=transcribe_data
        )
        transcribe_response.raise_for_status()
        prompt = transcribe_response.json().get("text", "")
        print(f"[DEBUG] Transcribed Text: {prompt}")

        # Step 2: ChatGPT
        chat_headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        chat_body = {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }

        chat_response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=chat_headers,
            json=chat_body
        )
        chat_response.raise_for_status()
        reply = chat_response.json()['choices'][0]['message']['content']
        print(f"[DEBUG] GPT-4o Response: {reply}")

        # Step 3: Text to Speech (TTS)
        tts_headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        tts_body = {
            "model": TTS_MODEL,
            "input": reply,
            "voice": TTS_VOICE
        }

        tts_response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers=tts_headers,
            json=tts_body
        )
        tts_response.raise_for_status()

        return send_file(
            io.BytesIO(tts_response.content),
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name="response.mp3"
        )

    except requests.exceptions.HTTPError as e:
        traceback.print_exc()
        return jsonify({"error": f"{e} | Response: {e.response.text}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
