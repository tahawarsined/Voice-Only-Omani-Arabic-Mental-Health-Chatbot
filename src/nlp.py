"""
Performs Natural Language Processing (NLP) tasks.

This module is responsible for analyzing the transcribed text to determine the user's
intent and emotional state. It uses a combination of keyword matching for critical
intents like 'crisis' and a pre-trained model from camel_tools for general
sentiment analysis.
"""

import logging
from typing import Dict, Literal

from camel_tools.sentiment import SentimentAnalyzer

from src.constants import CRISIS_KEYWORDS

logging.basicConfig(level=logging.INFO)

# Initialize the sentiment analyzer once
sa = SentimentAnalyzer.pretrained()

Intent = Literal["crisis", "seek_support", "casual", "unknown"]
Sentiment = Literal["positive", "negative", "neutral"]

def analyze_text(text: str) -> Dict[str, str]:
    """
    Analyzes the input text to determine intent and sentiment.

    Args:
        text: The transcribed text from the user.

    Returns:
        A dictionary containing the detected intent and sentiment.
    """
    if not text:
        return {"intent": "unknown", "sentiment": "neutral"}

    # 1. Intent Detection (Rule-based for crisis)
    intent: Intent = "seek_support"  # Default intent
    for keyword in CRISIS_KEYWORDS:
        if keyword in text.lower():
            intent = "crisis"
            logging.warning(f"Crisis keyword detected: '{keyword}'. Intent set to 'crisis'.")
            break

    # 2. Sentiment Analysis (using camel_tools)
    # The sentiment analyzer returns one of: 'positive', 'negative', 'neutral'
    sentiment: Sentiment = sa.predict([text])[0]
    logging.info(f"Text: '{text}' | Intent: {intent} | Sentiment: {sentiment}")

    return {"intent": intent, "sentiment": sentiment}

# Example usage (for testing purposes)
if __name__ == "__main__":
    test_cases = [
        "أنا أشعر بالحزن والوحدة",  # Sadness and loneliness
        "ما اريد اعيش خلاص",  # I don't want to live anymore (CRISIS)
        "شكراً لك، كلامك يريح",  # Thank you, your words are comforting
        "ايش اخبارك اليوم؟",  # How are you today? (Casual)
        "I think I need to kill myself immediately", # (CRISIS)
    ]

    for case in test_cases:
        analysis = analyze_text(case)
        print(f"'{case}' -> {analysis}")
