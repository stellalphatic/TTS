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
import urllib.request 

# Suppress excessive logging from libraries
logging.basicConfig(level=logging.INFO)
logging.getLogger('websockets.server').setLevel(logging.WARNING)
logging.getLogger('websockets.protocol').setLevel(logging.WARNING)
logging.getLogger('TTS.api').setLevel(logging.WARNING)
logging.getLogger('TTS.utils.io').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING) # Suppress urllib3 warnings
logging.getLogger('huggingface_hub').setLevel(logging.WARNING) # Suppress huggingface_hub warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING) # Suppress matplotlib font warnings

# --- Configuration ---
VOICE_SERVICE_SECRET_KEY = os.environ.get("VOICE_SERVICE_SECRET_KEY")
VOICE_SERVICE_PORT = int(os.environ.get("VOICE_SERVICE_PORT", 8765))
VOICE_SERVICE_HOST = os.environ.get("VOICE_SERVICE_HOST", "0.0.0.0")

# TTS_HOME will be set by the Dockerfile during build and available at runtime
TTS_MODEL_HOME = os.environ.get("TTS_HOME")
if not TTS_MODEL_HOME:
    logging.error("TTS_HOME environment variable not set. Model loading might fail.")
    # Fallback to default if not set, but pre-downloading is preferred.
    TTS_MODEL_HOME = os.path.expanduser("~/.local/share/tts")

# Define the full target directory path where model files are located
# This MUST match the full_cache_dir in download_model.py
TARGET_MODEL_DIR_SUFFIX = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_MODEL_FILES_DIR = os.path.join(TTS_MODEL_HOME, TARGET_MODEL_DIR_SUFFIX)

# Define explicit paths to the model files based on the actual download location
XTTS_MODEL_PATH = os.path.join(XTTS_MODEL_FILES_DIR, "model.pth")
XTTS_CONFIG_PATH = os.path.join(XTTS_MODEL_FILES_DIR, "config.json")

# --- Coqui TTS Model Loading ---
tts_model = None
speaker_embeddings = {} # Cache speaker embeddings for performance

async def load_tts_model():
    """Loads the Coqui TTS XTTS-v2 model globally."""
    global tts_model
    if tts_model is None:
        # Check if the expected model files exist
        if not os.path.exists(XTTS_MODEL_PATH) or not os.path.exists(XTTS_CONFIG_PATH):
            logging.error(f"Model files not found in expected directory: {XTTS_MODEL_FILES_DIR}")
            logging.error("Please ensure the Docker build successfully pre-downloaded the model.")
            raise FileNotFoundError(f"Missing XTTS-v2 model files in {XTTS_MODEL_FILES_DIR}")

        logging.info(f"Loading Coqui TTS XTTS-v2 model from {XTTS_MODEL_FILES_DIR}...")
        try:
            from TTS.api import TTS
            # Explicitly pass model_path and config_path to tell TTS where the files are
            # This should prevent it from trying to re-download or prompt.
            tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                            progress_bar=False,
                            gpu=True, # Ensure this is True if you have GPU enabled on Cloud Run
                            model_path=XTTS_MODEL_PATH,
                            config_path=XTTS_CONFIG_PATH)
            logging.info("Coqui TTS XTTS-v2 model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Coqui TTS model: {e}")
            tts_model = None
            raise

async def get_speaker_embedding(voice_clone_url, avatar_id):
    """Downloads voice sample and computes speaker embedding, caches it."""
    if avatar_id in speaker_embeddings:
        logging.info(f"Using cached speaker embedding for avatar {avatar_id}")
        return speaker_embeddings[avatar_id]

    logging.info(f"Downloading voice sample from: {voice_clone_url}")
    try:
        # Create a unique directory for each avatar's voice sample to avoid conflicts
        # This will be created within the main TTS_MODEL_HOME, not the nested model dir
        avatar_sample_dir = os.path.join(TTS_MODEL_HOME, "voice_samples", str(avatar_id))
        os.makedirs(avatar_sample_dir, exist_ok=True)

        file_extension = voice_clone_url.split('.')[-1].split('?')[0] # Handle query params in URL
        local_path = os.path.join(avatar_sample_dir, f"sample.{file_extension}")

        urllib.request.urlretrieve(voice_clone_url, local_path)

        logging.info(f"Voice sample downloaded to: {local_path}")

        speaker_embeddings[avatar_id] = local_path
        return local_path
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
        # Add padding back for base64urldecode if missing
        missing_padding = len(encoded_payload) % 4
        if missing_padding:
            encoded_payload += '=' * (4 - missing_padding)

        decoded_payload = base64.urlsafe_b64decode(encoded_payload).decode('utf-8')
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
                    for chunk in tts_model.tts_stream(
                        text=text,
                        speaker_wav=speaker_wav_path,
                        language="en" # Or detect language, or pass from Node.js
                    ):
                        # `chunk` is a numpy array (float32). Convert to bytes.
                        # Frontend needs to know sample rate (24kHz for XTTS-v2)
                        # and convert float32 to AudioBuffer.
                        await websocket.send(chunk.tobytes())

                    await websocket.send(json.dumps({"type": "speech_end"}))
                    logging.info("Finished streaming speech.")

                elif msg_type == 'stop_speaking':
                    logging.info("Received stop_speaking command. (Coqui TTS handles interruption internally if it's mid-stream)")
                    await websocket.send(json.dumps({"type": "speech_end"})) # Ensure frontend gets end signal

                else:
                    logging.warning(f"Received unknown message type: {msg_type}")

            except json.JSONDecodeError:
                logging.warning("Received non-JSON message from Node.js. Ignoring.")
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

async def main():
    """Main function to start the WebSocket server."""
    # Ensure the model is loaded before starting the server
    await load_tts_model()

    logging.info(f"Starting Python Voice Service on ws://{VOICE_SERVICE_HOST}:{VOICE_SERVICE_PORT}")
    server = await websockets.serve(
        voice_chat_websocket_handler,
        VOICE_SERVICE_HOST,
        VOICE_SERVICE_PORT,
        ping_interval=None, # Disable ping/pong for simplicity, can add if needed
        ping_timeout=None
    )
    await server.wait_closed()

if __name__ == "__main__":
    if not VOICE_SERVICE_SECRET_KEY:
        logging.error("VOICE_SERVICE_SECRET_KEY environment variable is not set. Exiting.")
        exit(1)

    if not TTS_MODEL_HOME:
        logging.warning("TTS_HOME environment variable not set. Using default user cache path.")

    asyncio.run(main())
