"""
Orchestrates the end-to-end voice chatbot pipeline.

This module integrates all the components (STT, NLP, Responder, TTS) into a
single, asynchronous pipeline. It manages the conversation state, handles the
flow of data from one stage to the next, and uses a ThreadPoolExecutor to run
blocking I/O operations (like API calls) without freezing the UI.

The main entry point is `process_audio_input`, which takes raw audio, processes it,
and returns the synthesized audio response.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

from src.constants import (
    CRISIS_RESPONSE_MESSAGE,
    CRISIS_SAFETY_CONFIRMATION_KEYWORDS,
)
from src.nlp import analyze_text
from src.responder import generate_response
from src.stt import SpeechToText  # We will use a simplified STT for Gradio
from src.tts import text_to_speech

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use a thread pool for concurrent execution of blocking tasks (LLM, TTS APIs)
executor = ThreadPoolExecutor(max_workers=5)

async def process_audio_input(
    audio_filepath: str,
    conversation_history: List[Dict[str, str]],
) -> Tuple[str, bytes, List[Dict[str, str]]]:
    """
    Processes a single turn of the conversation from audio input.

    Args:
        audio_filepath: The path to the temporary audio file from Gradio.
        conversation_history: The history of the conversation so far.

    Returns:
        A tuple containing:
        - The generated response text.
        - The synthesized audio response as bytes.
        - The updated conversation history.
    """
    loop = asyncio.get_running_loop()
    start_time = time.time()

    # 1. Speech-to-Text (STT)
    # In a real-time streaming app, this would be more complex.
    # For Gradio, we get a complete file, so we can use whisper directly.
    try:
        import whisper
        model = whisper.load_model("small")
        transcription_result = await loop.run_in_executor(
            executor, lambda: model.transcribe(audio_filepath, language="ar")
        )
        user_text = transcription_result["text"]
        logging.info(f"STT Transcribed: {user_text}")
    except Exception as e:
        logging.error(f"STT failed: {e}")
        user_text = ""

    stt_time = time.time()
    logging.info(f"STT finished in {stt_time - start_time:.2f}s")

    if not user_text:
        return "", b'', conversation_history

    # 2. NLP Analysis (Intent & Sentiment)
    nlp_result = analyze_text(user_text)
    intent = nlp_result["intent"]
    nlp_time = time.time()
    logging.info(f"NLP finished in {nlp_time - stt_time:.2f}s. Intent: {intent}")

    # 3. Crisis Intervention Logic
    # Check if the user is in a crisis state from a previous turn
    in_crisis_protocol = conversation_history and conversation_history[-1].get("context") == "crisis"

    if in_crisis_protocol:
        # If user confirms safety, exit crisis mode. Otherwise, repeat message.
        is_safe = any(word in user_text.lower() for word in CRISIS_SAFETY_CONFIRMATION_KEYWORDS)
        if is_safe:
            response_text = "شكراً لك. أنا هنا للمساعدة. كيف يمكنني دعمك الآن؟"
            conversation_history.append({"role": "user", "content": user_text})
        else:
            response_text = CRISIS_RESPONSE_MESSAGE # Repeat the crisis message
    elif intent == "crisis":
        response_text = CRISIS_RESPONSE_MESSAGE
        # Add context to know we are in the crisis protocol
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": response_text, "context": "crisis"})
    else:
        # 4. Generate Response (LLM)
        response_text = await loop.run_in_executor(
            executor, generate_response, conversation_history, user_text
        )
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": response_text})

    responder_time = time.time()
    logging.info(f"Responder finished in {responder_time - nlp_time:.2f}s")

    # 5. Text-to-Speech (TTS)
    audio_response = await loop.run_in_executor(executor, text_to_speech, response_text)
    tts_time = time.time()
    logging.info(f"TTS finished in {tts_time - responder_time:.2f}s")

    total_time = time.time() - start_time
    logging.info(f"--- Total turn time: {total_time:.2f}s ---")

    return response_text, audio_response, conversation_history
