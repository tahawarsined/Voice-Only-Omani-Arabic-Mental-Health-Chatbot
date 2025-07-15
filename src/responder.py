# ── src/responder.py ──────────────────────────────────────────────────────────
"""
Gemini wrapper (google-generativeai ≥0.8.x) that:
  • injects our SYSTEM_PROMPT as a prefix inside the very first user turn
  • converts stored turns to the only supported roles: 'user' and 'model'
"""

import google.generativeai as genai
from src.constants import GEMINI_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT

# One-time SDK setup
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)


def _gemini_msg(role: str, text: str) -> dict:
    """Return Gemini Content dict: {'role': role, 'parts': [text]}"""
    return {"role": role, "parts": [text]}


def generate_response(history, user_input: str) -> str:
    """
    history: list of {'role': 'user'|'assistant', 'content': str}
    user_input: latest user utterance (str)

    Returns Gemini's reply text.
    """
    gemini_history = []

    # 1️⃣  If this is the first turn → prepend SYSTEM_PROMPT to user_input
    first_turn = len(history) == 0
    if first_turn:
        user_input = f"{SYSTEM_PROMPT.strip()}\n\n{user_input}"

    # 2️⃣  Re-encode previous turns, mapping roles
    for msg in history:
        role = msg.get("role")
        text = msg.get("content", "")
        if not text:
            continue
        if role == "user":
            gemini_history.append(_gemini_msg("user", text))
        elif role == "assistant":
            gemini_history.append(_gemini_msg("model", text))
        # ignore anything else (e.g., 'system', internal markers)

    # 3️⃣  Start chat with that history
    chat = model.start_chat(history=gemini_history)

    # 4️⃣  Send latest user message
    resp = chat.send_message(user_input)   # role auto-set = 'user'

    # 5️⃣  Return plain text
    return resp.text.strip()
# ─────────────────────────────────────────────────────────────────────────────
