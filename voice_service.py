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

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


# Suppress excessive logging from libraries
logging.basicConfig(level=logging.INFO)
logging.getLogger('websockets.server').setLevel(logging.WARNING)
logging.getLogger('websockets.protocol').setLevel(logging.WARNING)
logging.getLogger('TTS.api').setLevel(logging.WARNING)
logging.getLogger('TTS.utils.io').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING) # Suppress urllib3 warnings
logging.getLogger('huggingface_hub').setLevel(logging.WARNING) # Suppress huggingface_hub warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING) # Suppress matplotlib font warnings
logging.getLogger('fsspec').setLevel(logging.WARNING) # Suppress fsspec warnings


# --- Configuration ---
VOICE_SERVICE_SECRET_KEY = os.environ.get("VOICE_SERVICE_SECRET_KEY")
VOICE_SERVICE_PORT = int(os.environ.get("VOICE_SERVICE_PORT", 8765))
VOICE_SERVICE_HOST = os.environ.get("VOICE_SERVICE_HOST", "0.0.0.0")

# TTS_HOME will be set by the Dockerfile during build and available at runtime
# This is also set as HF_HOME in the Dockerfile.
TTS_MODEL_HOME = os.environ.get("TTS_HOME")
if not TTS_MODEL_HOME:
    logging.error("TTS_HOME environment variable not set. Model loading might fail.")
    TTS_MODEL_HOME = os.path.expanduser("~/.local/share/tts")

# Define the full path to the directory containing the downloaded model files
# This is where download_model.py places the files (HF_HOME root)
XTTS_MODEL_DIR = os.path.join(TTS_MODEL_HOME, "models--coqui--XTTS-v2", "snapshots", os.listdir(os.path.join(TTS_MODEL_HOME, "models--coqui--XTTS-v2", "snapshots"))[0]) # This gets the latest snapshot hash dynamically

XTTS_CONFIG_PATH = os.path.join(XTTS_MODEL_DIR, "config.json")
XTTS_CHECKPOINT_DIR = XTTS_MODEL_DIR # checkpoint_dir is the directory containing model.pth, etc.

# --- Coqui TTS Model Loading ---
xtts_model = None
# Cache speaker latents (gpt_cond_latent, speaker_embedding) for performance
speaker_latents_cache = {}

async def load_tts_model():
    """Loads the Coqui XTTS model and configuration globally."""
    global xtts_model
    if xtts_model is None:
        logging.info(f"Loading Coqui XTTS-v2 model from {XTTS_MODEL_DIR}...")
        try:
            # 1. Load config
            config = XttsConfig()
            config.load_json(XTTS_CONFIG_PATH)

            # 2. Initialize model
            xtts_model = Xtts.init_from_config(config)

            # 3. Load checkpoint
            # use_deepspeed=False for now, as it adds complexity and might not be needed initially
            xtts_model.load_checkpoint(config, checkpoint_dir=XTTS_CHECKPOINT_DIR, use_deepspeed=False)

            # 4. Move model to GPU if available
            if torch.cuda.is_available():
                xtts_model.cuda()
                logging.info("Coqui XTTS-v2 model moved to CUDA (GPU).")
            else:
                logging.info("No CUDA (GPU) available, Coqui XTTS-v2 model loaded on CPU.")

            logging.info("Coqui XTTS-v2 model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Coqui XTTS model: {e}")
            xtts_model = None
            raise

async def get_speaker_latents(voice_clone_url, avatar_id):
    """Downloads voice sample and computes speaker latents, caches them."""
    if avatar_id in speaker_latents_cache:
        logging.info(f"Using cached speaker latents for avatar {avatar_id}")
        return speaker_latents_cache[avatar_id]

    logging.info(f"Downloading voice sample from: {voice_clone_url}")
    try:
        # Create a unique directory for each avatar's voice sample to avoid conflicts
        avatar_sample_dir = os.path.join(TTS_MODEL_HOME, "voice_samples", str(avatar_id))
        os.makedirs(avatar_sample_dir, exist_ok=True)

        file_extension = voice_clone_url.split('.')[-1].split('?')[0] # Handle query params in URL
        local_path = os.path.join(avatar_sample_dir, f"sample.{file_extension}")

        urllib.request.urlretrieve(voice_clone_url, local_path)

        logging.info(f"Voice sample downloaded to: {local_path}")

        # Compute speaker latents
        if xtts_model is None:
            raise RuntimeError("XTTS model not loaded to compute speaker latents.")

        # Ensure model is on the correct device for get_conditioning_latents
        device = "cuda" if torch.cuda.is_available() else "cpu"
        xtts_model.to(device)

        logging.info(f"Computing speaker latents for {avatar_id} on {device}...")
        gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(audio_path=[local_path])
        logging.info(f"Speaker latents computed for {avatar_id}.")

        speaker_latents_cache[avatar_id] = (gpt_cond_latent, speaker_embedding)
        return gpt_cond_latent, speaker_embedding
    except Exception as e:
        logging.error(f"Error downloading or processing voice sample/computing latents: {e}")
        return None, None

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
    gpt_cond_latent = None
    speaker_embedding = None

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

        gpt_cond_latent, speaker_embedding = await get_speaker_latents(voice_clone_url, avatar_id)
        if gpt_cond_latent is None or speaker_embedding is None:
            await websocket.send(json.dumps({"type": "error", "message": "Failed to load voice sample or compute latents."}))
            return

        if xtts_model is None:
            await websocket.send(json.dumps({"type": "error", "message": "XTTS model not loaded on server."}))
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

                    # Generate and stream audio chunks using inference_stream
                    chunks = xtts_model.inference_stream(
                        text=text,
                        language="en", # Or detect language, or pass from Node.js
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        # Add other inference parameters if needed, e.g., temperature, top_k, top_p
                    )

                    for chunk in chunks:
                        # `chunk` is a torch.Tensor. Convert to numpy and then bytes.
                        # Ensure it's on CPU before converting to numpy for sending.
                        await websocket.send(chunk.cpu().numpy().tobytes())

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
