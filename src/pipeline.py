# src/pipeline.py

import asyncio
import logging
from logging.handlers import RotatingFileHandler
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

from src.constants import (
    CRISIS_RESPONSE_MESSAGE,
    CRISIS_SAFETY_CONFIRMATION_KEYWORDS,
)
from src.nlp import analyze_text
from src.responder import generate_response
from src.tts import text_to_speech
from src.stt_whisper import transcribe

# ──────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP LOGGING ─ write DEBUG+ logs both to console and to a rolling file
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
logging.root.setLevel(logging.DEBUG)

# Console handler (INFO+)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter(LOG_FORMAT))
logging.root.addHandler(ch)

# File handler (DEBUG+, 5 MB per file, keep last 3)
fh = RotatingFileHandler("voicebot.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(LOG_FORMAT))
logging.root.addHandler(fh)
# ──────────────────────────────────────────────────────────────────────────────

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=5)

async def process_audio_input(
    audio_filepath: str,
    conversation_history: List[Dict[str, str]],
) -> Tuple[str, bytes, List[Dict[str, str]]]:
    """
    Processes a single turn of the conversation from audio input.

    Returns:
        - response_text: Generated reply text.
        - audio_response: Synthesized speech bytes.
        - updated conversation_history.
    """
    start_time = time.time()
    stt_time = None

    # 1. Speech-to-Text (STT)
    try:
        user_text = await asyncio.get_running_loop().run_in_executor(
            executor, transcribe, audio_filepath, "ar"
        )
        logging.info(f"STT Transcribed: '{user_text}'")
        stt_time = time.time()
        logging.debug(f"→ STT duration: {(stt_time - start_time):.2f}s")
    except Exception:
        logging.exception("STT failed")
        user_text = ""
        stt_time = time.time()
        logging.debug(f"→ STT duration (failure path): {(stt_time - start_time):.2f}s")
        # Bail out early on STT failure
        return "", b"", conversation_history

    # 2. NLP Analysis (Intent & Sentiment)
    nlp_time = time.time()
    if stt_time is None:
        logging.warning("stt_time was never set! Defaulting to start_time")
        stt_time = start_time
    nlp_result = analyze_text(user_text)
    intent = nlp_result["intent"]
    logging.info(f"NLP finished in {(nlp_time - stt_time):.2f}s. Intent: {intent}")

    # 3. Crisis Intervention Logic
    in_crisis_protocol = (
        conversation_history
        and conversation_history[-1].get("context") == "crisis"
    )

    if in_crisis_protocol:
        is_safe = any(
            word in user_text.lower()
            for word in CRISIS_SAFETY_CONFIRMATION_KEYWORDS
        )
        if is_safe:
            response_text = "شكراً لك. أنا هنا للمساعدة. كيف يمكنني دعمك الآن؟"
            conversation_history.append({"role": "user", "content": user_text})
        else:
            response_text = CRISIS_RESPONSE_MESSAGE
    elif intent == "crisis":
        response_text = CRISIS_RESPONSE_MESSAGE
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append(
            {"role": "assistant", "content": response_text, "context": "crisis"}
        )
    else:
        # 4. Generate Response (LLM)
        resp_start = time.time()
        response_text = await asyncio.get_running_loop().run_in_executor(
            executor, generate_response, conversation_history, user_text
        )
        resp_end = time.time()
        logging.info(f"Responder finished in {(resp_end - resp_start):.2f}s")
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": response_text})

    # 5. Text-to-Speech (TTS)
    tts_start = time.time()
    audio_response = await asyncio.get_running_loop().run_in_executor(
        executor, text_to_speech, response_text
    )
    tts_end = time.time()
    logging.info(f"TTS finished in {(tts_end - tts_start):.2f}s")

    total_time = time.time() - start_time
    logging.info(f"--- Total turn time: {total_time:.2f}s ---")

    return response_text, audio_response, conversation_history
