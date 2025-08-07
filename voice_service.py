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
from typing import Dict, Any, Tuple, AsyncGenerator, List, Union

from aiohttp import web, ClientSession, WSMsgType
import numpy as np
from scipy.io import wavfile
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import glob

# ==============================================================================
# --- 1. Configuration & Global State ---
# ==============================================================================

# Suppress excessive logging from libraries
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('websockets.server').setLevel(logging.WARNING)
logging.getLogger('websockets.protocol').setLevel(logging.WARNING)
logging.getLogger('TTS.api').setLevel(logging.WARNING)
logging.getLogger('TTS.utils.io').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('fsspec').setLevel(logging.WARNING)
logging.getLogger('aiohttp.access').setLevel(logging.WARNING)

# Environment variables
VOICE_SERVICE_SECRET_KEY = os.environ.get("VOICE_SERVICE_SECRET_KEY")
VOICE_SERVICE_PORT = int(os.environ.get("VOICE_SERVICE_PORT", 8765))
VOICE_SERVICE_HOST = os.environ.get("VOICE_SERVICE_HOST", "0.0.0.0")
TTS_MODEL_HOME = os.environ.get("TTS_HOME", os.path.expanduser("~/.local/share/tts"))

# Global TTS model and cache
xtts_model = None
xtts_config = None
speaker_latents_cache: Dict[str, Dict[str, Any]] = {}
SUPPORTED_LANGUAGES_XTTS = ['en', 'hi', 'es', 'fr', 'de', 'it', 'ja', 'ko', 'pt', 'ru']

# ==============================================================================
# --- 2. Core Service Functions ---
# ==============================================================================

def get_model_paths() -> Tuple[Union[str, None], Union[str, None]]:
    """
    Dynamically finds the XTTS model snapshot and config paths.
    Returns (config_path, checkpoint_dir) or (None, None) if not found.
    """
    xtts_model_base_dir = os.path.join(TTS_MODEL_HOME, "models--coqui--XTTS-v2", "snapshots")
    if not os.path.exists(xtts_model_base_dir):
        logging.error(f"XTTS model base directory not found: {xtts_model_base_dir}")
        return None, None

    # Use glob to find the snapshot directory robustly
    snapshot_dirs = glob.glob(os.path.join(xtts_model_base_dir, "*/"))
    if not snapshot_dirs:
        logging.error(f"XTTS model snapshot directory not found in {xtts_model_base_dir}. Ensure download_model.py ran.")
        return None, None

    xtts_model_dir = snapshot_dirs[0]
    xtts_config_path = os.path.join(xtts_model_dir, "config.json")
    return xtts_config_path, xtts_model_dir


async def load_tts_model():
    """Loads the Coqui XTTS model and configuration globally."""
    global xtts_model, xtts_config
    if xtts_model is not None:
        return

    logging.info("Attempting to load Coqui XTTS model...")
    xtts_config_path, xtts_checkpoint_dir = get_model_paths()
    if not xtts_config_path or not xtts_checkpoint_dir:
        raise RuntimeError("XTTS model paths not properly configured.")

    try:
        xtts_config = XttsConfig()
        xtts_config.load_json(xtts_config_path)

        xtts_model = Xtts.init_from_config(xtts_config)
        xtts_model.load_checkpoint(xtts_config, checkpoint_dir=xtts_checkpoint_dir, use_deepspeed=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        xtts_model.to(device)
        logging.info(f"Coqui XTTS-v2 model loaded successfully on {device}.")
    except Exception as e:
        logging.error(f"Failed to load Coqui XTTS model: {e}", exc_info=True)
        xtts_model = None
        xtts_config = None
        raise


async def get_speaker_latents(voice_clone_url: str, voice_id: str) -> Tuple[Union[Tuple, None], Union[str, None]]:
    """
    Downloads voice sample (if not cached), computes speaker latents, and caches them.
    This now uses aiohttp for non-blocking downloads.
    """
    if voice_id in speaker_latents_cache:
        logging.info(f"Using cached speaker latents for voice {voice_id}.")
        cached_data = speaker_latents_cache[voice_id]
        return cached_data['latents'], cached_data['local_path']

    logging.info(f"Downloading voice sample from: {voice_clone_url}")
    try:
        url_hash = hashlib.md5(voice_clone_url.encode()).hexdigest()
        voice_sample_dir = os.path.join(TTS_MODEL_HOME, "voice_samples", url_hash)
        os.makedirs(voice_sample_dir, exist_ok=True)

        file_extension = voice_clone_url.split('.')[-1].split('?')[0]
        local_path = os.path.join(voice_sample_dir, f"sample.{file_extension}")

        # Use aiohttp for non-blocking download
        if not os.path.exists(local_path):
            async with ClientSession() as session:
                async with session.get(voice_clone_url) as response:
                    if response.status != 200:
                        response.raise_for_status()
                    with open(local_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(1024)
                            if not chunk:
                                break
                            f.write(chunk)
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
        logging.error(f"Error downloading or processing voice sample: {e}", exc_info=True)
        return None, None


def verify_auth_token(token_header: str) -> bool:
    """Verifies the custom VOICE_CLONE_AUTH token with more specific error handling."""
    if not VOICE_SERVICE_SECRET_KEY:
        logging.error("VOICE_SERVICE_SECRET_KEY is not set.")
        return False
    if not token_header or not token_header.startswith("VOICE_CLONE_AUTH-"):
        logging.warning("Invalid token format received.")
        return False

    try:
        encoded_payload = token_header.split("VOICE_CLONE_AUTH-")[1]
        
        # Base64 requires padding, which may be missing
        missing_padding = len(encoded_payload) % 4
        if missing_padding:
            encoded_payload += '=' * (4 - missing_padding)

        decoded_payload = base64.urlsafe_b64decode(encoded_payload).decode('utf-8')
        signature, timestamp_str = decoded_payload.split('.')
        timestamp = int(timestamp_str)

        # Check for token expiration
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
    except (ValueError, IndexError, TypeError) as e:
        logging.error(f"Error parsing auth token: {e}", exc_info=True)
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during token verification: {e}", exc_info=True)
        return False


def validate_text_for_language(text: str, language: str) -> Tuple[bool, Union[str, None]]:
    """Performs basic validation of text content based on the selected language."""
    if language not in SUPPORTED_LANGUAGES_XTTS:
        return False, f"Unsupported language: {language}. Supported languages are: {', '.join(SUPPORTED_LANGUAGES_XTTS)}"

    if language == 'hi' and not any('\u0900' <= char <= '\u097F' for char in text):
        return False, "Text must contain Hindi (Devanagari) characters for Hindi language."
    return True, None


async def generate_audio_stream(
    text: str,
    language: str,
    local_voice_path: str,
    gpt_cond_latent,
    speaker_embedding
) -> AsyncGenerator[bytes, None]:
    """
    A reusable helper function to generate audio chunks and yield WAV formatted bytes.
    This replaces the duplicated generation logic in the handlers.
    """
    if xtts_model is None:
        raise RuntimeError("XTTS model is not loaded.")

    for chunk in xtts_model.inference_stream(
        text=text,
        language=language,
        speaker_wav=local_voice_path,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        stream_chunk_size=10
    ):
        audio_np = chunk.cpu().numpy()
        # Scale to 16-bit integer format and write to a WAV buffer
        audio_np_int16 = (audio_np * 32767).astype(np.int16)
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, 24000, audio_np_int16)
        yield wav_buffer.getvalue()


# ==============================================================================
# --- 3. aiohttp Handlers ---
# ==============================================================================

async def voice_chat_websocket_handler(request):
    """Handles WebSocket connections for real-time voice chat."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logging.info(f"New WebSocket connection from {request.remote}")

    # Authentication
    auth_token = request.headers.get('Authorization')
    if not auth_token or not verify_auth_token(auth_token):
        logging.error("WebSocket connection rejected: Invalid or missing token.")
        await ws.close(code=1008, reason="Authentication Failed.")
        return ws

    user_id, avatar_id, language, local_voice_path = None, None, 'en', None
    gpt_cond_latent, speaker_embedding = None, None

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                logging.warning(f"Received unexpected message type {msg.type}.")
                continue
            
            try:
                message = json.loads(msg.data)
                msg_type = message.get('type')

                if msg_type == 'init':
                    user_id = message.get('userId')
                    avatar_id = message.get('avatarId')
                    voice_clone_url = message.get('voice_clone_url')
                    language = message.get('language', 'en')
                    
                    if not all([user_id, avatar_id, voice_clone_url]):
                        await ws.send_json({"type": "error", "message": "Missing init parameters."})
                        break
                    
                    (latents, local_voice_path) = await get_speaker_latents(voice_clone_url, avatar_id)
                    if latents is None:
                        await ws.send_json({"type": "error", "message": "Failed to load voice sample."})
                        break
                    
                    gpt_cond_latent, speaker_embedding = latents
                    if xtts_model is None:
                        await ws.send_json({"type": "error", "message": "XTTS model not loaded."})
                        break

                    await ws.send_json({"type": "ready", "message": "Voice service ready."})
                    logging.info(f"Voice service ready for user {user_id}, avatar {avatar_id}.")

                elif msg_type == 'text_to_speak':
                    text = message.get('text')
                    if not text or not local_voice_path:
                        logging.warning("Received empty text or voice not initialized.")
                        continue
                    
                    await ws.send_json({"type": "speech_start"})
                    logging.info(f"Generating speech for text: '{text[:50]}...' in language '{language}'")
                    
                    async for wav_bytes in generate_audio_stream(text, language, local_voice_path, gpt_cond_latent, speaker_embedding):
                        await ws.send_bytes(wav_bytes)
                    
                    await ws.send_json({"type": "speech_end"})
                    logging.info("Finished streaming speech.")

                elif msg_type == 'stop_speaking':
                    logging.info("Received stop_speaking command.")
                    # Currently, we don't have a way to interrupt a stream from the outside.
                    await ws.send_json({"type": "speech_end"})

                else:
                    logging.warning(f"Received unknown message type: {msg_type}")

            except json.JSONDecodeError:
                logging.warning("Received non-JSON text message. Ignoring.")
            except Exception as e:
                logging.error(f"Error processing message: {e}", exc_info=True)
                await ws.send_json({"type": "error", "message": f"Server error: {e}"})
                break
    finally:
        logging.info(f"Cleaning up WebSocket connection for user {user_id}.")
        await ws.close()
    return ws


async def generate_audio_http_handler(request):
    """Handles HTTP POST requests for non-real-time audio generation."""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not verify_auth_token(auth_header):
        return web.json_response({"error": "Authentication Failed."}, status=401)
    
    try:
        data = await request.json()
        voice_id = data.get('voice_id')
        text = data.get('text')
        language = data.get('language', 'en')
        voice_clone_url = data.get('voice_clone_url')

        if not all([voice_id, text, text.strip(), voice_clone_url]):
            return web.json_response({"error": "Missing 'voice_id', 'text', or 'voice_clone_url'."}, status=400)
        
        is_valid_lang, lang_error_msg = validate_text_for_language(text, language)
        if not is_valid_lang:
            return web.json_response({"error": lang_error_msg}, status=400)
        
        (latents, local_voice_path) = await get_speaker_latents(voice_clone_url, voice_id)
        if latents is None or local_voice_path is None:
            return web.json_response({"error": "Failed to download voice sample or compute latents."}, status=500)
        
        gpt_cond_latent, speaker_embedding = latents
        logging.info(f"Generating non-streaming speech for voice {voice_id} (text: '{text[:50]}...')")

        full_audio_bytes = b''
        async for wav_bytes in generate_audio_stream(text, language, local_voice_path, gpt_cond_latent, speaker_embedding):
            full_audio_bytes += wav_bytes

        if not full_audio_bytes:
            logging.error("No audio bytes generated.")
            return web.json_response({"error": "Internal voice service error."}, status=500)
            
        return web.Response(body=full_audio_bytes, content_type='audio/wav')
    
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON in request body."}, status=400)
    except Exception as e:
        logging.error(f"Error in HTTP audio generation handler: {e}", exc_info=True)
        return web.json_response({"error": f"Internal server error: {e}"}, status=500)


async def video_stream_handler(request):
    """
    Handles WebSocket connections to bridge audio to a video service.
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
    
    user_id, avatar_id, language, local_voice_path = None, None, 'en', None
    gpt_cond_latent, speaker_embedding = None, None
    video_service_url = None

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT: continue
            message = json.loads(msg.data)
            msg_type = message.get('type')

            if msg_type == 'init':
                user_id = message.get('userId')
                avatar_id = message.get('avatarId')
                voice_clone_url = message.get('voice_clone_url')
                language = message.get('language', 'en')
                video_service_url = message.get('video_service_url')

                if not all([user_id, avatar_id, voice_clone_url, video_service_url]):
                    await ws.send_json({"type": "error", "message": "Missing init parameters for video stream."})
                    break

                (latents, local_voice_path) = await get_speaker_latents(voice_clone_url, avatar_id)
                if latents is None or xtts_model is None:
                    await ws.send_json({"type": "error", "message": "Failed to load voice sample or model."})
                    break
                gpt_cond_latent, speaker_embedding = latents

                await ws.send_json({"type": "ready", "message": "Voice service ready for video stream."})
                logging.info(f"Voice service ready for video stream for user {user_id}.")

            elif msg_type == 'text_to_speak':
                text = message.get('text')
                if not text or not local_voice_path or not video_service_url:
                    logging.error("Voice sample or video service not initialized.")
                    continue

                logging.info(f"Generating speech for video stream text: '{text[:50]}...'")
                
                async with ClientSession() as session:
                    try:
                        # Note: You should verify the SadTalker service's WebSocket endpoint path
                        video_ws = await session.ws_connect(f"{video_service_url}/real-time-stream/{avatar_id}")
                        logging.info(f"Connected to video service at {video_service_url}")
                        await video_ws.send_str(json.dumps({"type": "init", "avatarId": avatar_id}))

                        await ws.send_json({"type": "speech_start"})
                        
                        async for wav_bytes in generate_audio_stream(text, language, local_voice_path, gpt_cond_latent, speaker_embedding):
                            await video_ws.send_bytes(wav_bytes)
                        
                        await video_ws.close()
                        await ws.send_json({"type": "speech_end"})
                        logging.info("Finished streaming audio for video.")
                    except Exception as e:
                        logging.error(f"Failed to connect or stream to video service: {e}", exc_info=True)
                        await ws.send_json({"type": "error", "message": f"Failed to connect to video service: {e}"})

            elif msg_type == 'stop_speaking':
                logging.info("Received stop_speaking command.")
                await ws.send_json({"type": "speech_end"})

    except json.JSONDecodeError:
        logging.warning("Received non-JSON text message. Ignoring.")
    except Exception as e:
        logging.error(f"Unexpected error in video stream handler: {e}", exc_info=True)
    finally:
        logging.info(f"Cleaning up video stream connection for user {user_id}.")
        await ws.close()
    return ws


async def main():
    """Main function to start the aiohttp server."""
    if not VOICE_SERVICE_SECRET_KEY:
        logging.error("VOICE_SERVICE_SECRET_KEY is not set. Exiting.")
        exit(1)
    
    await load_tts_model()

    app = web.Application()
    app.router.add_post('/generate-audio', generate_audio_http_handler)
    app.router.add_get('/ws', voice_chat_websocket_handler)
    app.router.add_get('/ws-video-stream', video_stream_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, VOICE_SERVICE_HOST, VOICE_SERVICE_PORT)

    await site.start()
    logging.info(f"Python Voice Service running on http://{VOICE_SERVICE_HOST}:{VOICE_SERVICE_PORT}")

    await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Service shutting down.")
    except Exception as e:
        logging.error(f"An error occurred during application startup: {e}", exc_info=True)
        exit(1)
