"""
Handles real-time Speech-to-Text (STT) transcription.

This module provides a unified interface for transcribing audio streams using either
OpenAI Whisper or Vosk. It is designed to be run in a separate thread to avoid
blocking the main application loop, continuously listening to the microphone and
putting transcribed text into a queue.
"""

import logging
import queue
import threading
import time
from typing import Callable

import pyaudio
import whisper
from vosk import KaldiRecognizer, Model

from src.constants import (
    AUDIO_CHANNELS,
    AUDIO_CHUNK_SIZE,
    AUDIO_SAMPLE_RATE,
    STT_PROVIDER,
    VOSK_MODEL_PATH,
    WHISPER_MODEL,
)

logging.basicConfig(level=logging.INFO)

class SpeechToText:
    """A class to handle real-time speech-to-text transcription."""

    def __init__(self, on_transcription: Callable[[str], None]):
        """
        Initializes the STT engine based on the configured provider.

        Args:
            on_transcription: A callback function to be invoked with the final transcript.
        """
        self.on_transcription = on_transcription
        self.is_running = False
        self._thread = None
        self._audio_queue = queue.Queue()

        if STT_PROVIDER == "openai":
            logging.info(f"Loading OpenAI Whisper model: {WHISPER_MODEL}")
            self.model = whisper.load_model(WHISPER_MODEL)
        elif STT_PROVIDER == "vosk":
            logging.info(f"Loading Vosk model from: {VOSK_MODEL_PATH}")
            model = Model(VOSK_MODEL_PATH)
            self.recognizer = KaldiRecognizer(model, AUDIO_SAMPLE_RATE)
        else:
            raise ValueError(f"Unsupported STT provider: {STT_PROVIDER}")

    def _listen(self):
        """Continuously listens to the microphone and puts audio chunks into a queue."""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_SAMPLE_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK_SIZE,
        )

        logging.info("STT is listening...")
        while self.is_running:
            data = stream.read(AUDIO_CHUNK_SIZE)
            self._audio_queue.put(data)

        logging.info("STT stopped listening.")
        stream.stop_stream()
        stream.close()
        p.terminate()

    def _process(self):
        """Processes audio chunks from the queue and performs transcription."""
        while self.is_running:
            try:
                audio_data = self._audio_queue.get(timeout=1)
                if STT_PROVIDER == "openai":
                    # Whisper processes the entire audio at once after a pause
                    # This is a simplified implementation. A more robust one would detect speech segments.
                    # For this scaffold, we will assume a single utterance.
                    pass # Note: Whisper integration needs more sophisticated VAD.
                elif STT_PROVIDER == "vosk":
                    if self.recognizer.AcceptWaveform(audio_data):
                        result = self.recognizer.Result()
                        text = eval(result)["text"]
                        if text:
                            logging.info(f"Vosk transcribed: {text}")
                            self.on_transcription(text)
                    else:
                        partial_result = self.recognizer.PartialResult()
                        # logging.info(f"Partial: {partial_result}")

            except queue.Empty:
                continue

    def start(self):
        """Starts the listening and processing threads."""
        if not self.is_running:
            self.is_running = True
            self._listen_thread = threading.Thread(target=self._listen)
            self._process_thread = threading.Thread(target=self._process)
            self._listen_thread.start()
            self._process_thread.start()
            logging.info("STT service started.")

    def stop(self):
        """Stops the listening and processing threads."""
        if self.is_running:
            self.is_running = False
            self._listen_thread.join()
            self._process_thread.join()
            logging.info("STT service stopped.")

# Example usage (for testing purposes)
if __name__ == "__main__":

    def handle_transcription(text):
        print(f"Transcription received: {text}")

    stt = SpeechToText(on_transcription=handle_transcription)
    stt.start()

    try:
        print("Listening for 10 seconds... Speak into your microphone.")
        time.sleep(10)
    finally:
        stt.stop()
        print("STT test finished.")
