import asyncio
import json
import os
import logging
import hashlib
import hmac
import time
import base64
import io
import urllib.request

# NEW IMPORTS for aiohttp client and WebSocket
from aiohttp import web, ClientSession, WSMsgType

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
logging.getLogger('aiohttp.access').setLevel(logging.WARNING) # Suppress aiohttp access logs


# --- Configuration ---
VOICE_SERVICE_SECRET_KEY = os.environ.get("VOICE_SERVICE_SECRET_KEY")
VOICE_SERVICE_PORT = int(os.environ.get("PORT", 8765))
VOICE_SERVICE_HOST = os.environ.get("VOICE_SERVICE_HOST", "0.0.0.0")

TTS_MODEL_HOME = os.environ.get("TTS_HOME")
if not TTS_MODEL_HOME:
    logging.error("TTS_HOME environment variable not set. Model loading might fail.")
    TTS_MODEL_HOME = os.path.expanduser("~/.local/share/tts")

# Define the full path to the directory containing the downloaded model files
XTTS_MODEL_BASE_DIR = os.path.join(TTS_MODEL_HOME, "models--coqui--XTTS-v2", "snapshots")
try:
    XTTS_MODEL_DIR = os.path.join(XTTS_MODEL_BASE_DIR, os.listdir(XTTS_MODEL_BASE_DIR)[0])
    XTTS_CONFIG_PATH = os.path.join(XTTS_MODEL_DIR, "config.json")
    XTTS_CHECKPOINT_DIR = XTTS_MODEL_DIR
except IndexError:
    logging.error(f"XTTS model snapshot directory not found in {XTTS_MODEL_BASE_DIR}. Ensure download_model.py ran successfully.")
    XTTS_MODEL_DIR = None
    XTTS_CONFIG_PATH = None
    XTTS_CHECKPOINT_DIR = None


# --- Coqui TTS Model Loading ---
xtts_model = None
xtts_config = None # Make config globally accessible
speaker_latents_cache = {} # Cache for pre-computed latents (used by streaming)

async def load_tts_model():
    """Loads the Coqui XTTS model and configuration globally."""
    global xtts_model, xtts_config
    if xtts_model is None:
        if not XTTS_MODEL_DIR or not XTTS_CONFIG_PATH or not XTTS_CHECKPOINT_DIR:
            logging.error("XTTS model paths are not properly configured. Cannot load model.")
            raise RuntimeError("XTTS model paths not configured.")

        logging.info(f"Loading Coqui XTTS-v2 model from {XTTS_MODEL_DIR}...")
        try:
            xtts_config = XttsConfig()
            xtts_config.load_json(XTTS_CONFIG_PATH)

            xtts_model = Xtts.init_from_config(xtts_config)
            xtts_model.load_checkpoint(xtts_config, checkpoint_dir=XTTS_CHECKPOINT_DIR, use_deepspeed=False)

            if torch.cuda.is_available():
                xtts_model.cuda()
                logging.info("Coqui XTTS-v2 model moved to CUDA (GPU).")
            else:
                logging.info("No CUDA (GPU) available, Coqui XTTS-v2 model loaded on CPU.")

            logging.info("Coqui XTTS-v2 model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Coqui XTTS model: {e}")
            xtts_model = None
            xtts_config = None
            raise

async def get_speaker_latents(voice_clone_url, voice_id):
    """
    Downloads voice sample and computes speaker latents, caches them.
    Returns (gpt_cond_latent, speaker_embedding) and the local_path.
    """
    if voice_id in speaker_latents_cache:
        logging.info(f"Using cached speaker latents and path for voice {voice_id}")
        return speaker_latents_cache[voice_id]['latents'], speaker_latents_cache[voice_id]['local_path']

    logging.info(f"Downloading voice sample from: {voice_clone_url}")
    try:
        url_hash = hashlib.md5(voice_clone_url.encode()).hexdigest()
        voice_sample_dir = os.path.join(TTS_MODEL_HOME, "voice_samples", url_hash)
        os.makedirs(voice_sample_dir, exist_ok=True)

        file_extension = voice_clone_url.split('.')[-1].split('?')[0]
        local_path = os.path.join(voice_sample_dir, f"sample.{file_extension}")

        if not os.path.exists(local_path):
            urllib.request.urlretrieve(voice_clone_url, local_path)
            logging.info(f"Voice sample downloaded to: {local_path}")
        else:
            logging.info(f"Voice sample already exists at: {local_path}")

        if xtts_model is None:
            raise RuntimeError("XTTS model not loaded to compute speaker latents.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        xtts_model.to(device)

        logging.info(f"Computing speaker latents for {voice_id} on {device}...")
        gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(audio_path=[local_path])
        logging.info(f"Speaker latents computed for {voice_id}.")

        speaker_latents_cache[voice_id] = {'latents': (gpt_cond_latent, speaker_embedding), 'local_path': local_path}
        return (gpt_cond_latent, speaker_embedding), local_path
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
SUPPORTED_LANGUAGES_XTTS = ['en', 'hi', 'es', 'fr', 'de', 'it', 'ja', 'ko', 'pt', 'ru'] # XTTS supported languages

def validate_text_for_language(text, language):
    """
    Performs basic validation of text content based on the selected language.
    """
    if language not in SUPPORTED_LANGUAGES_XTTS:
        return False, f"Unsupported language: {language}. Supported languages are: {', '.join(SUPPORTED_LANGUAGES_XTTS)}"

    if language == 'hi':
        hindi_char_present = any('\u0900' <= char <= '\u097F' for char in text)
        if not hindi_char_present:
            return False, "Text must contain Hindi (Devanagari) characters for Hindi language."
    return True, None

# --- WebSocket Handler (Existing Real-time Chat) ---
async def voice_chat_websocket_handler(request):
    """Handles incoming WebSocket connections for real-time voice chat using aiohttp."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    logging.info(f"New WebSocket connection from {request.remote}")

    auth_token = request.headers.get('Authorization')
    if not auth_token: # Check for header existence
        logging.error("WebSocket connection rejected: Missing Authorization header.")
        await ws.close(code=1008, reason="Authentication Failed: Missing Authorization header.")
        return ws

    if not verify_auth_token(auth_token):
        logging.error("WebSocket connection rejected: Invalid authentication token.")
        await ws.close(code=1008, reason="Authentication Failed: Invalid token.")
        return ws # Return the WebSocketResponse object

    logging.info("WebSocket connection authenticated.")

    user_id = None
    avatar_id = None
    gpt_cond_latent = None
    speaker_embedding = None
    language = 'en'
    local_voice_path = None # Store the local path for WebSocket streaming

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    message = json.loads(msg.data)
                    msg_type = message.get('type')

                    if msg_type == 'init':
                        user_id = message.get('userId')
                        avatar_id = message.get('avatarId')
                        voice_clone_url = message.get('voice_clone_url')
                        language = message.get('language', 'en')

                        if not all([user_id, avatar_id, voice_clone_url]):
                            logging.error("Missing init parameters. Closing connection.")
                            await ws.send_json({"type": "error", "message": "Missing userId, avatarId, or voice_clone_url in init."})
                            break # Exit message loop

                        logging.info(f"Initializing voice for user {user_id}, avatar {avatar_id} from {voice_clone_url} in language {language}")

                        (gpt_cond_latent, speaker_embedding), local_voice_path = await get_speaker_latents(voice_clone_url, avatar_id)
                        if gpt_cond_latent is None or speaker_embedding is None or local_voice_path is None:
                            await ws.send_json({"type": "error", "message": "Failed to load voice sample or compute latents."})
                            break # Exit message loop

                        if xtts_model is None:
                            await ws.send_json({"type": "error", "message": "XTTS model not loaded on server."})
                            break # Exit message loop

                        await ws.send_json({"type": "ready", "message": "Voice service ready."})
                        logging.info(f"Voice service ready for user {user_id}, avatar {avatar_id}")

                    elif msg_type == 'text_to_speak':
                        text = message.get('text')
                        if not text:
                            logging.warning("Received empty text_to_speak message.")
                            continue
                        if local_voice_path is None:
                            logging.error("Voice sample not initialized for streaming. Cannot generate speech.")
                            await ws.send_json({"type": "error", "message": "Voice sample not initialized for streaming."})
                            continue

                        logging.info(f"Generating speech for text: '{text[:50]}...' in language '{language}'")
                        await ws.send_json({"type": "speech_start"})

                        for chunk in xtts_model.inference_stream(
                            text=text,
                            language=language,
                            speaker_wav=local_voice_path,
                            gpt_cond_latent=gpt_cond_latent,
                            speaker_embedding=speaker_embedding,
                        ):
                            audio_np = chunk.cpu().numpy()
                            audio_np_int16 = (audio_np * 32767).astype(np.int16)
                            wav_buffer = io.BytesIO()
                            wavfile.write(wav_buffer, 24000, audio_np_int16)
                            wav_bytes = wav_buffer.getvalue()
                            await ws.send_bytes(wav_bytes)

                        await ws.send_json({"type": "speech_end"})
                        logging.info("Finished streaming speech.")

                    elif msg_type == 'stop_speaking':
                        logging.info("Received stop_speaking command. (Coqui TTS handles interruption internally if it's mid-stream)")
                        await ws.send_json({"type": "speech_end"})

                    else:
                        logging.warning(f"Received unknown message type: {msg_type}")

                except json.JSONDecodeError:
                    logging.warning("Received non-JSON text message from Node.js. Ignoring.")
                except Exception as e:
                    logging.error(f"Error processing message from Node.js: {e}", exc_info=True)
                    await ws.send_json({"type": "error", "message": f"Server error: {e}"})
            elif msg.type == WSMsgType.BINARY:
                logging.warning("Received unexpected binary message on WebSocket.")
            elif msg.type == WSMsgType.ERROR:
                logging.error(f"WebSocket connection closed with error: {ws.exception()}")
            elif msg.type == WSMsgType.CLOSE:
                logging.info("WebSocket connection closed by client.")
                break
    except Exception as e:
        logging.error(f"Unexpected error in voice chat handler loop: {e}", exc_info=True)
    finally:
        logging.info(f"Cleaning up WebSocket connection for user {user_id}.")
        await ws.close()
    return ws


# --- HTTP Handler (For Non-Real-time Audio Generation) ---
async def generate_audio_http_handler(request):
    """
    Handles HTTP POST requests to generate audio from text using a specific voice.
    """
    logging.info(f"New HTTP request to /generate-audio from {request.remote}")

    auth_header = request.headers.get('Authorization')
    if not auth_header:
        logging.error("HTTP request rejected: Missing Authorization header.")
        return web.json_response({"error": "Authentication Failed: Missing Authorization header."}, status=401)

    if not verify_auth_token(auth_header):
        logging.error("HTTP request rejected: Invalid authentication token.")
        return web.json_response({"error": "Authentication Failed: Invalid token."}, status=401)

    try:
        data = await request.json()
        voice_id = data.get('voice_id')
        text = data.get('text')
        language = data.get('language', 'en')
        voice_clone_url = data.get('voice_clone_url')

        if not all([voice_id, text, text.strip(), voice_clone_url]):
            return web.json_response({"error": "Missing 'voice_id', 'text', or 'voice_clone_url' in request body."}, status=400)

        is_valid_lang, lang_error_msg = validate_text_for_language(text, language)
        if not is_valid_lang:
            return web.json_response({"error": lang_error_msg}, status=400)

        (gpt_cond_latent, speaker_embedding), local_voice_path = await get_speaker_latents(voice_clone_url, voice_id)
        if gpt_cond_latent is None or speaker_embedding is None or local_voice_path is None:
            return web.json_response({"error": "Failed to download voice sample or compute latents. Check voice_clone_url validity."}, status=500)

        if xtts_model is None:
            return web.json_response({"error": "XTTS model not loaded on server."}, status=500)

        logging.info(f"Generating non-streaming speech for voice {voice_id} (text: '{text[:50]}...') in language {language} using inference_stream")

        audio_chunks = []
        for chunk in xtts_model.inference_stream(
            text=text,
            language=language,
            speaker_wav=local_voice_path,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
        ):
            audio_chunks.append(chunk.cpu().numpy())

        if not audio_chunks:
            logging.error("inference_stream yielded no audio chunks.")
            return web.json_response({"error": "Internal voice service error: No audio chunks generated."}, status=500)

        full_audio_np = np.concatenate(audio_chunks)

        audio_np_int16 = (full_audio_np * 32767).astype(np.int16)
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, 24000, audio_np_int16)
        wav_bytes = wav_buffer.getvalue()

        logging.info(f"Successfully generated audio for voice {voice_id}.")
        return web.Response(body=wav_bytes, content_type='audio/wav')

    except json.JSONDecodeError:
        logging.error("Invalid JSON in HTTP request body.")
        return web.json_response({"error": "Invalid JSON in request body."}, status=400)
    except Exception as e:
        logging.error(f"Error in HTTP audio generation handler: {e}", exc_info=True)
        return web.json_response({"error": f"Internal server error: {e}"}, status=500)


# --- NEW: WebSocket Handler for Real-time Video Stream Audio ---
async def video_stream_handler(request):
    """
    Handles incoming WebSocket connections to bridge audio to the video service.
    This endpoint takes text, generates audio, and forwards the audio chunks
    to the video generation service via a separate WebSocket.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logging.info(f"New WebSocket connection for video stream from {request.remote}")

    # Authentication
    auth_token = request.headers.get('Authorization')
    if not auth_token or not verify_auth_token(auth_token):
        logging.error("Video stream connection rejected: Invalid or missing token.")
        await ws.close(code=1008, reason="Authentication Failed.")
        return ws

    user_id = None
    avatar_id = None
    gpt_cond_latent = None
    speaker_embedding = None
    language = 'en'
    local_voice_path = None
    video_service_url = None

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                message = json.loads(msg.data)
                msg_type = message.get('type')

                if msg_type == 'init':
                    user_id = message.get('userId')
                    avatar_id = message.get('avatarId')
                    voice_clone_url = message.get('voice_clone_url')
                    language = message.get('language', 'en')
                    video_service_url = message.get('video_service_url') # The URL of the SadTalker service

                    if not all([user_id, avatar_id, voice_clone_url, video_service_url]):
                        logging.error("Missing init parameters for video stream. Closing.")
                        await ws.send_json({"type": "error", "message": "Missing init params."})
                        break

                    logging.info(f"Initializing voice for video stream: user {user_id}, avatar {avatar_id}")
                    (gpt_cond_latent, speaker_embedding), local_voice_path = await get_speaker_latents(voice_clone_url, avatar_id)
                    if gpt_cond_latent is None or speaker_embedding is None or local_voice_path is None:
                        await ws.send_json({"type": "error", "message": "Failed to load voice sample."})
                        break

                    if xtts_model is None:
                        await ws.send_json({"type": "error", "message": "XTTS model not loaded."})
                        break

                    await ws.send_json({"type": "ready", "message": "Voice service ready for video stream."})
                    logging.info(f"Voice service ready for video stream for user {user_id}.")

                elif msg_type == 'text_to_speak':
                    text = message.get('text')
                    if not text: continue
                    if local_voice_path is None or video_service_url is None:
                        logging.error("Voice sample or video service not initialized.")
                        continue

                    logging.info(f"Generating speech for video stream text: '{text[:50]}...'")

                    # Connect to the SadTalker video service
                    async with ClientSession() as session:
                        try:
                            video_ws = await session.ws_connect(f"{video_service_url}/real-time-stream/{avatar_id}")
                            logging.info(f"Connected to SadTalker video service at {video_service_url}")
                            await video_ws.send_str(json.dumps({"type": "init", "avatarId": avatar_id}))

                            await ws.send_json({"type": "speech_start"}) # Notify client that speech generation has begun
                            
                            # Stream audio to the video service
                            for chunk in xtts_model.inference_stream(
                                text=text,
                                language=language,
                                speaker_wav=local_voice_path,
                                gpt_cond_latent=gpt_cond_latent,
                                speaker_embedding=speaker_embedding,
                            ):
                                audio_np = chunk.cpu().numpy()
                                audio_np_int16 = (audio_np * 32767).astype(np.int16)
                                wav_buffer = io.BytesIO()
                                wavfile.write(wav_buffer, 24000, audio_np_int16)
                                wav_bytes = wav_buffer.getvalue()
                                
                                # Forward the audio bytes to the SadTalker service
                                await video_ws.send_bytes(wav_bytes)

                            await video_ws.close()
                            await ws.send_json({"type": "speech_end"})
                            logging.info("Finished streaming audio for video.")

                        except Exception as e:
                            logging.error(f"Failed to connect or stream to video service: {e}", exc_info=True)
                            await ws.send_json({"type": "error", "message": f"Failed to connect to video service: {e}"})

                elif msg_type == 'stop_speaking':
                    logging.info("Received stop_speaking command.")
                    # Acknowledge the stop command, the connection to the video service will be closed in the try block
                    await ws.send_json({"type": "speech_end"})

                else:
                    logging.warning(f"Received unknown message type: {msg_type}")

    except Exception as e:
        logging.error(f"Unexpected error in video stream handler: {e}", exc_info=True)
    finally:
        logging.info(f"Cleaning up video stream connection for user {user_id}.")
        await ws.close()
    return ws


# --- Main Application Setup ---
async def main():
    """Main function to start the aiohttp server handling both HTTP and WebSocket."""
    await load_tts_model()

    app = web.Application()
    
    # HTTP route for audio generation
    app.router.add_post('/generate-audio', generate_audio_http_handler)
    
    # WebSocket route for real-time voice chat
    app.router.add_get('/ws', voice_chat_websocket_handler) # Existing
    
    # NEW: WebSocket route for video streaming audio
    app.router.add_get('/ws-video-stream', video_stream_handler) # New

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, VOICE_SERVICE_HOST, VOICE_SERVICE_PORT)

    await site.start()
    logging.info(f"Python Voice Service running on http://{VOICE_SERVICE_HOST}:{VOICE_SERVICE_PORT} (HTTP), ws://.../ws, and ws://.../ws-video-stream")

    await asyncio.Future()

if __name__ == "__main__":
    if not VOICE_SERVICE_SECRET_KEY:
        logging.error("VOICE_SERVICE_SECRET_KEY environment variable is not set. Exiting.")
        exit(1)

    if not TTS_MODEL_HOME:
        logging.warning("TTS_HOME environment variable not set. Using default user cache path.")

    asyncio.run(main())