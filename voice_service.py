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
from aiohttp import web # NEW: For HTTP server

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# NEW IMPORTS for WAV conversion
import numpy as np
from scipy.io import wavfile

# Suppress excessive logging from libraries
logging.basicConfig(level=logging.INFO)
logging.getLogger('websockets.server').setLevel(logging.WARNING)
logging.getLogger('websockets.protocol').setLevel(logging.WARNING)
logging.getLogger('TTS.api').setLevel(logging.WARNING)
logging.getLogger('TTS.utils.io').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('fsspec').setLevel(logging.WARNING)


# --- Configuration ---
VOICE_SERVICE_SECRET_KEY = os.environ.get("VOICE_SERVICE_SECRET_KEY")
VOICE_SERVICE_PORT = int(os.environ.get("VOICE_SERVICE_PORT", 8765))
VOICE_SERVICE_HOST = os.environ.get("VOICE_SERVICE_HOST", "0.0.0.0")

TTS_MODEL_HOME = os.environ.get("TTS_HOME")
if not TTS_MODEL_HOME:
    logging.error("TTS_HOME environment variable not set. Model loading might fail.")
    TTS_MODEL_HOME = os.path.expanduser("~/.local/share/tts")

# Define the full path to the directory containing the downloaded model files
XTTS_MODEL_DIR = os.path.join(TTS_MODEL_HOME, "models--coqui--XTTS-v2", "snapshots", os.listdir(os.path.join(TTS_MODEL_HOME, "models--coqui--XTTS-v2", "snapshots"))[0])
XTTS_CONFIG_PATH = os.path.join(XTTS_MODEL_DIR, "config.json") # Corrected path
XTTS_CHECKPOINT_DIR = XTTS_MODEL_DIR # checkpoint_dir is the directory containing model.pth, etc.


# --- Coqui TTS Model Loading ---
xtts_model = None
speaker_latents_cache = {}

async def load_tts_model():
    """Loads the Coqui XTTS model and configuration globally."""
    global xtts_model
    if xtts_model is None:
        logging.info(f"Loading Coqui XTTS-v2 model from {XTTS_MODEL_DIR}...")
        try:
            config = XttsConfig()
            config.load_json(XTTS_CONFIG_PATH)

            xtts_model = Xtts.init_from_config(config)
            xtts_model.load_checkpoint(config, checkpoint_dir=XTTS_CHECKPOINT_DIR, use_deepspeed=False)

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

async def get_speaker_latents(voice_clone_url, voice_id): # Changed avatar_id to voice_id for clarity
    """Downloads voice sample and computes speaker latents, caches them."""
    if voice_id in speaker_latents_cache:
        logging.info(f"Using cached speaker latents for voice {voice_id}")
        return speaker_latents_cache[voice_id]

    logging.info(f"Downloading voice sample from: {voice_clone_url}")
    try:
        # Create a unique directory for each voice sample
        voice_sample_dir = os.path.join(TTS_MODEL_HOME, "voice_samples", str(voice_id))
        os.makedirs(voice_sample_dir, exist_ok=True)

        file_extension = voice_clone_url.split('.')[-1].split('?')[0]
        local_path = os.path.join(voice_sample_dir, f"sample.{file_extension}")

        urllib.request.urlretrieve(voice_clone_url, local_path)

        logging.info(f"Voice sample downloaded to: {local_path}")

        if xtts_model is None:
            raise RuntimeError("XTTS model not loaded to compute speaker latents.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        xtts_model.to(device)

        logging.info(f"Computing speaker latents for {voice_id} on {device}...")
        gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(audio_path=[local_path])
        logging.info(f"Speaker latents computed for {voice_id}.")

        speaker_latents_cache[voice_id] = (gpt_cond_latent, speaker_embedding)
        return gpt_cond_latent, speaker_embedding
    except Exception as e:
        logging.error(f"Error downloading or processing voice sample/computing latents: {e}")
        return None, None

# --- Authentication ---
def verify_auth_token(token_header):
    """Verifies the custom VOICE_CLONE_AUTH token."""
    if not token_header or not token_header.startswith("VOICE_CLONE_AUTH-"):
        logging.warning("Invalid token format received.")
        return False

    try:
        encoded_payload = token_header.split("VOICE_CLONE_AUTH-")[1]
        missing_padding = len(encoded_payload) % 4
        if missing_padding:
            encoded_payload += '=' * (4 - missing_padding)

        decoded_payload = base64.urlsafe_b64decode(encoded_payload).decode('utf-8')
        signature, timestamp_str = decoded_payload.split('.')
        timestamp = int(timestamp_str)

        if abs(time.time() - timestamp) > 300: # 300 seconds = 5 minutes
            logging.warning(f"Auth token expired or too old. Timestamp: {timestamp}, Current: {time.time()}")
            return False

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

# --- Language Validation (for HTTP endpoint) ---
def validate_text_for_language(text, language):
    """
    Performs basic validation of text content based on the selected language.
    This is a simple regex-based check and can be expanded for more robustness.
    """
    if language == 'hi':
        # Check for presence of Devanagari script characters
        hindi_regex = r'[\u0900-\u097F]'
        if not any(char for char in text if '\u0900' <= char <= '\u097F'):
            return False, "Text must contain Hindi (Devanagari) characters for Hindi language."
    elif language == 'en':
        # Basic check for non-English characters if English is strictly enforced
        # For XTTS, it's generally flexible, but if you want strict validation:
        # english_regex = r'^[a-zA-Z0-9\s.,!?;:\-_\'"()]+$'
        # if not re.match(english_regex, text):
        #     return False, "Text must be in English for English language."
        pass # No strict enforcement for English for now
    # Add more language-specific checks as needed
    return True, None

# --- WebSocket Handler (Existing Real-time Chat) ---
async def voice_chat_websocket_handler(websocket, path):
    """Handles incoming WebSocket connections for real-time voice chat."""
    logging.info(f"New WebSocket connection from {websocket.remote_address}")

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
    language = 'en'

    try:
        init_message_raw = await websocket.recv()
        init_message = json.loads(init_message_raw)

        if init_message.get('type') != 'init':
            logging.error("First message was not 'init'. Closing connection.")
            await websocket.send(json.dumps({"type": "error", "message": "Expected 'init' message first."}))
            return

        user_id = init_message.get('userId')
        avatar_id = init_message.get('avatarId')
        voice_clone_url = init_message.get('voice_clone_url')
        language = init_message.get('language', 'en')

        if not all([user_id, avatar_id, voice_clone_url]):
            logging.error("Missing init parameters. Closing connection.")
            await websocket.send(json.dumps({"type": "error", "message": "Missing userId, avatarId, or voice_clone_url in init."}))
            return

        logging.info(f"Initializing voice for user {user_id}, avatar {avatar_id} from {voice_clone_url} in language {language}")

        # Use avatar_id as voice_id for caching speaker latents in real-time chat
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

                    logging.info(f"Generating speech for text: '{text[:50]}...' in language '{language}'")
                    await websocket.send(json.dumps({"type": "speech_start"}))

                    for chunk in xtts_model.inference_stream(
                        text=text,
                        language=language,
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                    ):
                        audio_np = chunk.cpu().numpy()
                        audio_np_int16 = (audio_np * 32767).astype(np.int16)

                        wav_buffer = io.BytesIO()
                        wavfile.write(wav_buffer, 24000, audio_np_int16) # XTTS-v2 typically outputs at 24000 Hz
                        wav_bytes = wav_buffer.getvalue()

                        await websocket.send(wav_bytes)

                    await websocket.send(json.dumps({"type": "speech_end"}))
                    logging.info("Finished streaming speech.")

                elif msg_type == 'stop_speaking':
                    logging.info("Received stop_speaking command. (Coqui TTS handles interruption internally if it's mid-stream)")
                    await websocket.send(json.dumps({"type": "speech_end"}))

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


# --- HTTP Handler (NEW: For Non-Real-time Audio Generation) ---
async def generate_audio_http_handler(request):
    """
    Handles HTTP POST requests to generate audio from text using a specific voice.
    Expects JSON payload: {"voice_id": "uuid", "text": "string", "language": "string"}
    Returns WAV audio file.
    """
    logging.info(f"New HTTP request to /generate-audio from {request.remote}")

    # Authentication for HTTP endpoint (reusing the same secret key logic)
    auth_header = request.headers.get('Authorization')
    if not verify_auth_token(auth_header):
        logging.error("HTTP request rejected: Invalid or missing authentication token.")
        return web.json_response({"error": "Authentication Failed"}, status=401)

    try:
        data = await request.json()
        voice_id = data.get('voice_id')
        text = data.get('text')
        language = data.get('language', 'en') # Default to English

        if not all([voice_id, text, text.strip()]):
            return web.json_response({"error": "Missing 'voice_id' or 'text' in request body."}, status=400)

        is_valid_lang, lang_error_msg = validate_text_for_language(text, language)
        if not is_valid_lang:
            return web.json_response({"error": lang_error_msg}, status=400)

        # Retrieve voice URL from Supabase (via backend service cache)
        # The backend Node.js service will fetch the voice_url from Supabase and pass it here.
        # So, this voice_service assumes voice_clone_url is directly provided if voice_id is given.
        # This means the Node.js backend needs to fetch the audio_url from the 'voices' table
        # and pass it as 'voice_clone_url' in the HTTP request to this service.
        voice_clone_url = data.get('voice_clone_url') # Expect Node.js backend to provide this

        if not voice_clone_url:
            logging.error(f"Voice clone URL not provided for voice_id: {voice_id}. Cannot generate audio.")
            return web.json_response({"error": "Voice clone URL not provided by backend."}, status=400)

        gpt_cond_latent, speaker_embedding = await get_speaker_latents(voice_clone_url, voice_id)
        if gpt_cond_latent is None or speaker_embedding is None:
            return web.json_response({"error": "Failed to load voice sample or compute latents."}, status=500)

        if xtts_model is None:
            return web.json_response({"error": "XTTS model not loaded on server."}, status=500)

        logging.info(f"Generating non-streaming speech for voice {voice_id} (text: '{text[:50]}...') in language {language}")

        # Generate audio (non-streaming)
        # xtts_model.generate_speech returns a numpy array
        audio_array = xtts_model.generate_speech(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            # Add other inference parameters if needed
        )

        # Convert float32 numpy array to int16 WAV bytes
        audio_np_int16 = (audio_array * 32767).astype(np.int16)
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, 24000, audio_np_int16) # XTTS-v2 typically outputs at 24000 Hz
        wav_bytes = wav_buffer.getvalue()

        logging.info(f"Successfully generated audio for voice {voice_id}.")
        return web.Response(body=wav_bytes, content_type='audio/wav')

    except json.JSONDecodeError:
        logging.error("Invalid JSON in HTTP request body.")
        return web.json_response({"error": "Invalid JSON in request body."}, status=400)
    except Exception as e:
        logging.error(f"Error in HTTP audio generation handler: {e}", exc_info=True)
        return web.json_response({"error": f"Internal server error: {e}"}, status=500)


# --- Main Application Setup ---
async def main():
    """Main function to start both WebSocket and HTTP servers."""
    await load_tts_model()

    # Setup HTTP application
    app = web.Application()
    app.router.add_post('/generate-audio', generate_audio_http_handler) # NEW HTTP endpoint

    # Create a runner for the aiohttp app
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, VOICE_SERVICE_HOST, VOICE_SERVICE_PORT)

    # Start the HTTP server
    await site.start()
    logging.info(f"HTTP server running on http://{VOICE_SERVICE_HOST}:{VOICE_SERVICE_PORT}/generate-audio")

    # Start the WebSocket server
    ws_server = await websockets.serve(
        voice_chat_websocket_handler,
        VOICE_SERVICE_HOST,
        VOICE_SERVICE_PORT,
        ping_interval=None,
        ping_timeout=None
    )
    logging.info(f"WebSocket server running on ws://{VOICE_SERVICE_HOST}:{VOICE_SERVICE_PORT}")

    # Keep both servers running indefinitely
    await asyncio.Future() # This will keep the event loop running

if __name__ == "__main__":
    if not VOICE_SERVICE_SECRET_KEY:
        logging.error("VOICE_SERVICE_SECRET_KEY environment variable is not set. Exiting.")
        exit(1)

    if not TTS_MODEL_HOME:
        logging.warning("TTS_HOME environment variable not set. Using default user cache path.")

    asyncio.run(main())
