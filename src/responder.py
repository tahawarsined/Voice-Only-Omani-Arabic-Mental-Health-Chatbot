"""
Generates therapeutic responses using a dual-LLM strategy.

This module interfaces with OpenAI (primary) and Anthropic (fallback/validator)
models to generate empathetic and context-aware replies. It constructs a prompt
that includes the system persona, conversation history, and the latest user input,
then calls the appropriate LLM.

It includes a fallback mechanism to ensure a response is always generated, even if
the primary LLM fails.
"""
import logging
from typing import List, Dict

import anthropic
import openai

from src.constants import (
    ANTHROPIC_API_KEY,
    FALLBACK_LLM,
    OPENAI_API_KEY,
    PRIMARY_LLM,
    SYSTEM_PROMPT,
)

logging.basicConfig(level=logging.INFO)

# Initialize API clients
if not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
    raise ValueError("API keys for OpenAI and Anthropic must be set in .env file.")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def generate_response(
    conversation_history: List[Dict[str, str]], user_input: str
) -> str:
    """
    Generates a response using the primary LLM, with a fallback to the secondary LLM.

    Args:
        conversation_history: A list of previous turns in the conversation.
        user_input: The latest transcribed text from the user.

    Returns:
        The generated text response from the LLM.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": user_input},
    ]

    try:
        logging.info(f"Calling primary LLM ({PRIMARY_LLM}) with user input: {user_input}")
        response = openai_client.chat.completions.create(
            model=PRIMARY_LLM,
            messages=messages,
            temperature=0.7,  # Encourage empathetic but not overly creative responses
        )
        main_response = response.choices[0].message.content.strip()
        logging.info(f"Primary LLM generated response: {main_response}")
        return main_response
    except Exception as e:
        logging.error(f"Primary LLM ({PRIMARY_LLM}) failed: {e}. Attempting fallback.")
        try:
            logging.info(f"Calling fallback LLM ({FALLBACK_LLM}) with user input: {user_input}")
            # Anthropic uses a different message format
            anthropic_messages = messages[1:] # Remove system prompt from messages list
            response = anthropic_client.messages.create(
                model=FALLBACK_LLM,
                system=SYSTEM_PROMPT,
                max_tokens=1024,
                messages=anthropic_messages
            )
            fallback_response = response.content[0].text.strip()
            logging.info(f"Fallback LLM generated response: {fallback_response}")
            return fallback_response
        except Exception as e_fallback:
            logging.error(f"Fallback LLM ({FALLBACK_LLM}) also failed: {e_fallback}")
            return "عذراً، أواجه صعوبة في الاتصال الآن. هل يمكنك المحاولة مرة أخرى؟" # Generic error message

# Example usage (for testing purposes)
if __name__ == "__main__":
    # This test requires API keys to be set in a .env file
    if not all([OPENAI_API_KEY, ANTHROPIC_API_KEY]):
        print("Skipping responder test: Please set OPENAI_API_KEY and ANTHROPIC_API_KEY in a .env file.")
    else:
        print("--- Testing Normal Response ---")
        history = []
        user_text = "أشعر ببعض القلق اليوم"
        reply = generate_response(history, user_text)
        print(f"User: {user_text}")
        print(f"Bot: {reply}")

        print("\n--- Testing Follow-up Response ---")
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        user_text_2 = "لا أعرف السبب، كل شيء يبدو جيداً"
        reply_2 = generate_response(history, user_text_2)
        print(f"User: {user_text_2}")
        print(f"Bot: {reply_2}")