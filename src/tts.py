
"""
Handles Text-to-Speech (TTS) synthesis.

This module converts the chatbot's text responses into audible speech. It provides a
unified interface for generating audio using either Microsoft Azure's Neural TTS
(primary, high-quality) or gTTS (fallback, offline).

The generated audio is returned as bytes, which can then be played back directly.
"""


import logging
import os
from io import BytesIO

import azure.cognitiveservices.speech as speechsdk
from gtts import gTTS

from src.constants import (
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    AZURE_VOICE_NAME_FEMALE,
    GTTS_LANG,
    TTS_PROVIDER,
)

logging.basicConfig(level=logging.INFO)

def _get_azure_speech_synthesizer() -> speechsdk.SpeechSynthesizer:
    """Configures and returns an Azure SpeechSynthesizer instance."""
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        raise ValueError("Azure Speech key and region must be set in .env file.")
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = AZURE_VOICE_NAME_FEMALE
    # Use an in-memory stream for audio output
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False, filename=None)
    return speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

def text_to_speech(text: str) -> bytes:
    """
    Synthesizes speech from text using the configured TTS provider.

    Args:
        text: The text to be converted to speech.

    Returns:
        The raw audio data as bytes.
    """
    logging.info(f"Synthesizing speech for text: '{text[:30]}...' ")
    audio_data = b''

    if TTS_PROVIDER == "azure":
        try:
            synthesizer = _get_azure_speech_synthesizer()
            result = synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_data = result.audio_data
                logging.info("Azure TTS synthesis successful.")
            else:
                logging.error(f"Azure TTS error: {result.reason} - {result.cancellation_details}")
                # Fallback to gTTS if Azure fails
                return text_to_speech_gtts(text)
        except Exception as e:
            logging.error(f"Azure TTS failed: {e}. Falling back to gTTS.")
            return text_to_speech_gtts(text)

    elif TTS_PROVIDER == "gtts":
        audio_data = text_to_speech_gtts(text)
    else:
        raise ValueError(f"Unsupported TTS provider: {TTS_PROVIDER}")

    return audio_data

def text_to_speech_gtts(text: str) -> bytes:
    """Generates speech using gTTS as a fallback."""
    logging.info("Using gTTS for speech synthesis.")
    try:
        tts = gTTS(text=text, lang=GTTS_LANG, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        logging.error(f"gTTS failed: {e}")
        return b'' # Return empty bytes if all synthesis fails

# Example usage (for testing purposes)
if __name__ == "__main__":
    test_text = "مرحباً، كيف يمكنني مساعدتك اليوم؟"

    # Test Azure (requires credentials)
    if AZURE_SPEECH_KEY and AZURE_SPEECH_REGION:
        print("--- Testing Azure TTS ---")
        audio = text_to_speech(test_text)
        if audio:
            with open("output_azure.wav", "wb") as f:
                f.write(audio)
            print("Saved Azure TTS output to output_azure.wav")
        else:
            print("Azure TTS failed, but fallback should have been attempted.")
    else:
        print("Skipping Azure TTS test: Credentials not found in .env file.")

    # Test gTTS directly
    print("\n--- Testing gTTS Fallback ---")
    audio_gtts = text_to_speech_gtts(test_text)
    if audio_gtts:
        with open("output_gtts.mp3", "wb") as f:
            f.write(audio_gtts)
        print("Saved gTTS output to output_gtts.mp3")
    else:
        print("gTTS failed.")

