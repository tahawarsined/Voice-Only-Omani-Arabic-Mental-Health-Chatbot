"""
Constants and configuration values for the Omani-Therapist-Voice chatbot.

This module centralizes settings like API keys, model names, and crisis intervention
details to make the application easier to configure and maintain. It loads sensitive
information from a .env file for security.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys and Credentials ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

# --- Speech-to-Text (STT) Configuration ---
STT_PROVIDER = "openai"  # "openai" or "vosk"
WHISPER_MODEL = "small"  # Recommended model for performance/accuracy balance
VOSK_MODEL_PATH = "models/vosk-model-ar-0.22-linto-1.1.0"  # Path to Vosk model

# --- Natural Language Processing (NLP) Configuration ---
# Keywords to detect crisis intent. Case-insensitive.
CRISIS_KEYWORDS = [
    "انتحار", "suicide", "اقتل نفسي", "kill myself", "موت", "death",
    "اذي نفسي", "harm myself", "وحيد", "alone", "يأس", "hopeless",
    "ما اريد اعيش", "don't want to live"
]

# --- Responder (LLM) Configuration ---
PRIMARY_LLM = "gpt-4o"
FALLBACK_LLM = "claude-3-opus-20240229"
# System prompt to guide the LLM's behavior
SYSTEM_PROMPT = """
You are an empathetic and culturally-aware Omani mental health chatbot.
Your role is to provide supportive conversations based on Cognitive Behavioral Therapy (CBT) principles.
- **Language**: Respond in Omani Arabic. You can understand English code-switching.
- **Cultural Sensitivity**: Be respectful of Islamic values, Gulf family norms, and Omani culture. Use local idioms where appropriate.
- **Therapeutic Approach**: Use active listening, validation, and gentle questioning. Focus on helping the user identify and challenge negative thought patterns.
- **DO NOT**: Give direct advice, diagnose conditions, or act as a replacement for a human therapist.
- **Crisis**: If the user expresses suicidal thoughts, immediately provide the crisis hotlines and do not deviate from the crisis protocol.
"""

# --- Text-to-Speech (TTS) Configuration ---
TTS_PROVIDER = "azure"  # "azure" or "gtts"
AZURE_VOICE_NAME_FEMALE = "ar-OM-AyshaNeural"
AZURE_VOICE_NAME_MALE = "ar-OM-AbdullahNeural"
GTTS_LANG = "ar"

# --- Crisis Intervention ---
OMAN_CRISIS_HOTLINE = "+968 24 607 555"
OMAN_EMERGENCY_NUMBER = "9999"
CRISIS_RESPONSE_MESSAGE = f"""
أسمع في كلامك ألم كبير وحزن عميق. أريدك أن تعرف أنك لست وحدك, وهناك من يريد مساعدتك.
إذا كنت تفكر في إيذاء نفسك, أرجوك تواصل مع أحد هذه الأرقام فوراً:
- خط الدعم النفسي في عمان: {OMAN_CRISIS_HOTLINE}
- رقم الطوارئ: {OMAN_EMERGENCY_NUMBER}
أنا هنا لأسمعك, هل يمكنك أن تعدني بأنك ستتصل بهم الآن?
"""
CRISIS_SAFETY_CONFIRMATION_KEYWORDS = ["نعم", "yes", "ok", "حاضر", "زين", "بتصل"]

# --- Audio Settings ---
AUDIO_FORMAT = "wav"
AUDIO_SAMPLE_RATE = 16000  # Hz
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SIZE = 1024  # Samples