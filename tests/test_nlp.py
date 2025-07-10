"""
Unit tests for the NLP module.

This file contains tests to ensure the reliability of the NLP analysis, with a
particular focus on the critical function of crisis intent detection. It uses
standard Python `unittest` to define test cases.
"""

import unittest
import sys
import os

# Add the src directory to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.nlp import analyze_text

class TestNlp(unittest.TestCase):
    """Test suite for NLP functions."""

    def test_crisis_intent_arabic(self):
        """Tests if Arabic crisis keywords are correctly identified."""
        crisis_phrases = [
            "أفكر في الانتحار",
            "اريد اقتل نفسي خلاص",
            "الحياة ما لها طعم وأفكر في الموت",
            "أشعر برغبة في إيذاء نفسي",
        ]
        for phrase in crisis_phrases:
            with self.subTest(phrase=phrase):
                result = analyze_text(phrase)
                self.assertEqual(result["intent"], "crisis")

    def test_crisis_intent_english(self):
        """Tests if English crisis keywords are correctly identified."""
        crisis_phrases = [
            "I want to commit suicide",
            "I am going to kill myself",
            "I feel so hopeless and want to die",
            "I have a plan to harm myself",
        ]
        for phrase in crisis_phrases:
            with self.subTest(phrase=phrase):
                result = analyze_text(phrase)
                self.assertEqual(result["intent"], "crisis")

    def test_non_crisis_intent(self):
        """Tests that normal phrases are not misclassified as a crisis."""
        normal_phrases = [
            "أشعر بالحزن اليوم",
            "مرحباً، كيف حالك؟",
            "شكراً على مساعدتك",
            "I'm feeling a bit down today",
        ]
        for phrase in normal_phrases:
            with self.subTest(phrase=phrase):
                result = analyze_text(phrase)
                self.assertNotEqual(result["intent"], "crisis")
                self.assertEqual(result["intent"], "seek_support")

    def test_sentiment_analysis(self):
        """Tests the sentiment analysis functionality."""
        test_cases = {
            "أنا سعيد جداً اليوم": "positive",
            "أشعر بالأسف الشديد": "negative",
            "هذا هو الحال": "neutral",
            "I am very happy": "positive",
            "This is really bad": "negative",
        }
        for text, expected_sentiment in test_cases.items():
            with self.subTest(text=text):
                result = analyze_text(text)
                self.assertEqual(result["sentiment"], expected_sentiment)

    def test_empty_input(self):
        """Tests the behavior with empty or whitespace input."""
        result = analyze_text("")
        self.assertEqual(result["intent"], "unknown")
        self.assertEqual(result["sentiment"], "neutral")
        result_ws = analyze_text("   ")
        self.assertEqual(result_ws["intent"], "seek_support") # camel_tools might classify whitespace

if __name__ == "__main__":
    unittest.main()
