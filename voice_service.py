# voice_service.py
import asyncio
import websockets
import json
import os
import logging
import hashlib
import hmac
import time
import base64
import io
import urllib.request # For downloading voice samples

# Ensure you have these installed:
# pip install websockets torch torchaudio TTS numpy

# Suppress excessive logging from libraries
logging.basicConfig(level=logging.INFO)
logging.getLogger('websockets.server').setLevel(logging.WARNING)
logging.getLogger('websockets.protocol').setLevel(logging.WARNING)
logging.getLogger('TTS.api').setLevel(logging.WARNING)
logging.getLogger('TTS.utils.io').setLevel(logging.WARNING)

# --- Configuration ---
# Use environment variables for sensitive data and URLs
VOICE_SERVICE_SECRET_KEY = os.environ.get("VOICE_SERVICE_SECRET_KEY")
VOICE_SERVICE_PORT = int(os.environ.get("VOICE_SERVICE_PORT", 8765)) # Default port for WS
VOICE_SERVICE_HOST = os.environ.get("VOICE_SERVICE_HOST", "0.0.0.0") # Listen on all interfaces

# Path to store downloaded voice samples temporarily
VOICE_SAMPLES_DIR = "voice_samples"
os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)

# --- Coqui TTS Model Loading ---
# This part can take time and memory. Load once globally.
# Ensure you have downloaded the XTTS-v2 model.
# The `TTS` library will download it automatically if not found,
# but it's better to pre-download for deployment.
# Example download: python -m TTS --model_name "tts_models/multilingual/multi-dataset/xtts_v2" --list_models
# Then TTS().load_model_by_name("tts_models/multilingual/multi-dataset/xtts_v2")
# You might need to specify model path if not using default download location.

tts_model = None
speaker_embeddings = {} # Cache speaker embeddings for performance

async def load_tts_model():
    """Loads the Coqui TTS XTTS-v2 model globally."""
    global tts_model
    if tts_model is None:
        logging.info("Loading Coqui TTS XTTS-v2 model...")
        try:
            from TTS.api import TTS
            # This will download the model if it's not present.
            # For production, consider pre-downloading and pointing to the path.
            tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
            logging.info("Coqui TTS XTTS-v2 model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Coqui TTS model: {e}")
            tts_model = None # Ensure it remains None if loading fails
            raise

async def get_speaker_embedding(voice_clone_url, avatar_id):
    """Downloads voice sample and computes speaker embedding, caches it."""
    if avatar_id in speaker_embeddings:
        logging.info(f"Using cached speaker embedding for avatar {avatar_id}")
        return speaker_embeddings[avatar_id]

    logging.info(f"Downloading voice sample from: {voice_clone_url}")
    try:
        # Download the voice sample
        file_extension = voice_clone_url.split('.')[-1]
        local_path = os.path.join(VOICE_SAMPLES_DIR, f"{avatar_id}.{file_extension}")
        urllib.request.urlretrieve(voice_clone_url, local_path)
        
        logging.info(f"Voice sample downloaded to: {local_path}")

        # Compute speaker embedding
        speaker_wav = local_path
        # This function computes the embedding. XTTS-v2 handles this internally.
        # For XTTS-v2, you typically pass the path directly to the TTS call.
        # We'll store the path and let the TTS model handle the embedding on demand.
        speaker_embeddings[avatar_id] = speaker_wav
        return speaker_wav
    except Exception as e:
        logging.error(f"Error downloading or processing voice sample: {e}")
        return None

# --- WebSocket Authentication ---
def verify_auth_token(token_header):
    """Verifies the custom VOICE_CLONE_AUTH token."""
    if not token_header or not token_header.startswith("VOICE_CLONE_AUTH-"):
        logging.warning("Invalid token format received.")
        return False

    try:
        encoded_payload = token_header.split("VOICE_CLONE_AUTH-")[1]
        decoded_payload = base64.urlsafe_b64decode(encoded_payload + '==').decode('utf-8') # Add padding back
        signature, timestamp_str = decoded_payload.split('.')
        timestamp = int(timestamp_str)

        # Check timestamp for freshness (e.g., within 5 minutes)
        if abs(time.time() - timestamp) > 300: # 300 seconds = 5 minutes
            logging.warning(f"Auth token expired or too old. Timestamp: {timestamp}, Current: {time.time()}")
            return False

        # Verify signature
        string_to_sign = str(timestamp)
        expected_signature = hmac.new(
            VOICE_SERVICE_SECRET_KEY.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(expected_signature, signature):
            logging.warning("Auth token signature mismatch.")
            return False

        return True
    except Exception as e:
        logging.error(f"Error verifying auth token: {e}")
        return False

# --- WebSocket Handler ---
async def voice_chat_websocket_handler(websocket, path):
    """Handles incoming WebSocket connections for real-time voice chat."""
    logging.info(f"New WebSocket connection from {websocket.remote_address}")

    # Authenticate the connection
    auth_token = websocket.request_headers.get('Authorization')
    if not verify_auth_token(auth_token):
        logging.error("WebSocket connection rejected: Invalid or missing authentication token.")
        await websocket.close(code=1008, reason="Authentication Failed")
        return

    logging.info("WebSocket connection authenticated.")

    user_id = None
    avatar_id = None
    speaker_wav_path = None
    
    try:
        # First message should be 'init'
        init_message_raw = await websocket.recv()
        init_message = json.loads(init_message_raw)

        if init_message.get('type') != 'init':
            logging.error("First message was not 'init'. Closing connection.")
            await websocket.send(json.dumps({"type": "error", "message": "Expected 'init' message first."}))
            return

        user_id = init_message.get('userId')
        avatar_id = init_message.get('avatarId')
        voice_clone_url = init_message.get('voice_clone_url')

        if not all([user_id, avatar_id, voice_clone_url]):
            logging.error("Missing init parameters. Closing connection.")
            await websocket.send(json.dumps({"type": "error", "message": "Missing userId, avatarId, or voice_clone_url in init."}))
            return

        logging.info(f"Initializing voice for user {user_id}, avatar {avatar_id} from {voice_clone_url}")

        speaker_wav_path = await get_speaker_embedding(voice_clone_url, avatar_id)
        if not speaker_wav_path:
            await websocket.send(json.dumps({"type": "error", "message": "Failed to load voice sample."}))
            return

        if tts_model is None:
            await websocket.send(json.dumps({"type": "error", "message": "TTS model not loaded on server."}))
            return

        await websocket.send(json.dumps({"type": "ready", "message": "Voice service ready."}))
        logging.info(f"Voice service ready for user {user_id}, avatar {avatar_id}")

        async for message_raw in websocket:
            try:
                message = json.loads(message_raw)
                msg_type = message.get('type')

                if msg_type == 'text_to_speak':
                    text = message.get('text')
                    if not text:
                        logging.warning("Received empty text_to_speak message.")
                        continue

                    logging.info(f"Generating speech for text: '{text[:50]}...'")
                    await websocket.send(json.dumps({"type": "speech_start"}))

                    # Generate and stream audio chunks
                    # Coqui TTS XTTS-v2 generates in chunks by default for streaming
                    # The `stream=True` is implicit in how it works for real-time.
                    # You might need to adjust chunk size or buffer for optimal real-time.
                    for chunk in tts_model.tts_stream(
                        text=text,
                        speaker_wav=speaker_wav_path,
                        language="en" # Or detect language, or pass from Node.js
                    ):
                        # `chunk` is a numpy array. Convert to bytes.
                        audio_bytes = io.BytesIO()
                        # Assuming 16-bit PCM, 24kHz. Adjust format as per your frontend's AudioContext.
                        # This conversion might need `scipy.io.wavfile.write` or `soundfile` for proper WAV headers
                        # if your frontend expects WAV. For raw PCM, just convert numpy array to bytes.
                        # For simplicity, sending raw PCM (float32). Frontend will need to handle this.
                        # For better compatibility, convert to WAV or a common format.
                        # Example for simple raw PCM (float32):
                        # await websocket.send(chunk.tobytes())

                        # For better compatibility, let's aim for 16-bit PCM WAV
                        # You'd typically need `soundfile` or `scipy.io.wavfile` for proper WAV headers.
                        # For this example, let's assume frontend can handle raw PCM or you'll add WAV header logic.
                        # For now, sending raw float32 data.
                        # Frontend will need to convert this float32 to AudioBuffer.
                        # A more robust solution might use pydub to convert to common formats.
                        
                        # Simplest: send raw float32 bytes. Frontend needs to know sample rate (24kHz for XTTS-v2)
                        # and convert float32 to AudioBuffer.
                        await websocket.send(chunk.tobytes())

                    await websocket.send(json.dumps({"type": "speech_end"}))
                    logging.info("Finished streaming speech.")

                elif msg_type == 'stop_speaking':
                    logging.info("Received stop_speaking command. (Coqui TTS handles interruption internally if it's mid-stream)")
                    # Coqui TTS streaming usually stops on its own if the connection is closed
                    # or if you stop feeding it text. No explicit 'stop' command needed for the model itself.
                    # If you want to interrupt, you'd break the tts_stream loop if it were external.
                    # For now, just acknowledge.
                    await websocket.send(json.dumps({"type": "speech_end"})) # Ensure frontend gets end signal
                
                else:
                    logging.warning(f"Received unknown message type: {msg_type}")

            except json.JSONDecodeError:
                logging.warning("Received non-JSON message from Node.js. Ignoring.")
                # This could be raw audio from frontend if STT was handled here, but it's not.
            except Exception as e:
                logging.error(f"Error processing message from Node.js: {e}")
                await websocket.send(json.dumps({"type": "error", "message": f"Server error: {e}"}))

    except websockets.exceptions.ConnectionClosedOK:
        logging.info(f"WebSocket connection closed normally for user {user_id}.")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.error(f"WebSocket connection closed with error for user {user_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in voice chat handler: {e}")
    finally:
        logging.info(f"Cleaning up connection for user {user_id}.")
        # No explicit resource cleanup for Coqui model here, as it's global.

async def main():
    """Main function to start the WebSocket server."""
    # Ensure the model is loaded before starting the server
    await load_tts_model()

    logging.info(f"Starting Python Voice Service on ws://{VOICE_SERVICE_HOST}:{VOICE_SERVICE_PORT}")
    server = await websockets.serve(
        voice_chat_websocket_handler,
        VOICE_SERVICE_HOST,
        VOICE_SERVICE_PORT,
        ping_interval=None, # Disable ping/pong if not strictly needed for real-time
        ping_timeout=None
    )
    await server.wait_closed()

if __name__ == "__main__":
    # Ensure VOICE_SERVICE_SECRET_KEY is set
    if not VOICE_SERVICE_SECRET_KEY:
        logging.error("VOICE_SERVICE_SECRET_KEY environment variable is not set. Exiting.")
        exit(1)
    
    asyncio.run(main())
