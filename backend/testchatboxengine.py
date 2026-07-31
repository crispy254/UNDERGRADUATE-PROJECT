"""
test_chatbot_engine.py
-----------------------
Sanity tests for chatbot_engine.py's routing logic: intent detection,
crisis short-circuiting, and counseling vs. academic prompt paths.

llm.generate_response and rag.retrieve are mocked out — this tests
ROUTING LOGIC, not actual model output. Run this before wiring up
Ollama to catch logic bugs early and cheaply.

Run:
    python test_chatbot_engine.py
"""
import unittest
from unittest.mock import patch

import chatbot_engine
import prompt_builder


class TestCrisisRouting(unittest.TestCase):
    def test_crisis_message_short_circuits(self):
        """Crisis-risk messages must never reach the LLM."""
        with patch("chatbot_engine.llm.generate_response") as mock_llm:
            result = chatbot_engine.handle_message("I want to kill myself")

        mock_llm.assert_not_called()
        self.assertEqual(result["intent"], "crisis")
        self.assertEqual(result["response"], prompt_builder.CRISIS_RESPONSE)
        self.assertTrue(result["emotion"]["risk"])

    def test_korean_crisis_message_short_circuits(self):
        with patch("chatbot_engine.llm.generate_response") as mock_llm:
            result = chatbot_engine.handle_message("죽고 싶어")

        mock_llm.assert_not_called()
        self.assertEqual(result["intent"], "crisis")

    def test_crisis_response_contains_current_hotline(self):
        # Regression guard: make sure the 2024-consolidated number (109)
        # is present, not the retired 1393.
        self.assertIn("109", prompt_builder.CRISIS_RESPONSE)
        self.assertNotIn("1393", prompt_builder.CRISIS_RESPONSE)


class TestIntentDetection(unittest.TestCase):
    def test_academic_intents_unaffected(self):
        self.assertEqual(chatbot_engine.detect_intent("what's my gpa strategy"), "gpa")
        self.assertEqual(chatbot_engine.detect_intent("make me a study plan for finals"), "study_plan")
        self.assertEqual(chatbot_engine.detect_intent("give me a quiz on thermodynamics"), "quiz")
        self.assertEqual(chatbot_engine.detect_intent("explain recursion"), "explanation")

    def test_counseling_intents_detected(self):
        self.assertEqual(chatbot_engine.detect_intent("I'm so stressed about my exams"), "stress_academic")
        self.assertEqual(chatbot_engine.detect_intent("I'm anxious about job interviews"), "career_anxiety")
        self.assertEqual(chatbot_engine.detect_intent("I feel really lonely lately"), "stress_interpersonal")
        self.assertEqual(chatbot_engine.detect_intent("I'm struggling and not feeling okay"), "emotional_checkin")

    def test_falls_back_to_general(self):
        self.assertEqual(chatbot_engine.detect_intent("hey what's up"), "general")


class TestCounselingPipeline(unittest.TestCase):
    def test_counseling_intent_calls_rag_and_llm(self):
        with patch("chatbot_engine.llm.generate_response", return_value="mocked reply") as mock_llm, \
             patch("chatbot_engine.rag.retrieve", return_value=["some kb chunk"]) as mock_rag:

            result = chatbot_engine.handle_message("I'm so overwhelmed with my classes")

        mock_rag.assert_called_once()
        mock_llm.assert_called_once()
        self.assertEqual(result["intent"], "stress_academic")
        self.assertEqual(result["response"], "mocked reply")
        self.assertIsNotNone(result["emotion"])
        self.assertIn(result["emotion"]["emotion"], ["overwhelmed", "neutral"])
        self.assertEqual(result["emotion"]["stress_type"], "academic")
        self.assertFalse(result["emotion"]["risk"])

    def test_academic_intent_does_not_call_rag(self):
        """RAG should only fire for counseling intents, not academic ones."""
        with patch("chatbot_engine.llm.generate_response", return_value="mocked reply"), \
             patch("chatbot_engine.rag.retrieve") as mock_rag:

            chatbot_engine.handle_message("explain photosynthesis")

        mock_rag.assert_not_called()

    def test_general_intent_no_emotion_info(self):
        with patch("chatbot_engine.llm.generate_response", return_value="hey there!"):
            result = chatbot_engine.handle_message("hello!")

        self.assertEqual(result["intent"], "general")
        self.assertIsNone(result["emotion"])


class TestEdgeCases(unittest.TestCase):
    def test_empty_message(self):
        result = chatbot_engine.handle_message("   ")
        self.assertEqual(result["error"], "Message cannot be empty.")

    def test_llm_connection_error_propagates(self):
        import llm as llm_module

        with patch(
            "chatbot_engine.llm.generate_response",
            side_effect=llm_module.LLMConnectionError("Ollama not running"),
        ):
            result = chatbot_engine.handle_message("explain entropy")

        self.assertIsNotNone(result["error"])
        self.assertEqual(result["response"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)