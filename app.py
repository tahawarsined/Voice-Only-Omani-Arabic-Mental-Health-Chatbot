"""
Main application entry point for the Gradio UI.

This script launches the user interface for the Omani-Therapist-Voice chatbot.
It uses the Gradio library to create a simple, accessible interface with a
microphone input for the user to speak and an audio output for the chatbot's
voice response.

The conversation state, including the history, is maintained within the Gradio
session to ensure context is preserved across multiple turns.
"""

import logging
import asyncio

import gradio as gr

from src.pipeline import process_audio_input

logging.basicConfig(level=logging.INFO)

def voice_chat(audio_input, history_state):
    """
    Gradio interface function. It takes audio input and conversation history,
    processes them through the pipeline, and returns the updated UI components.

    Args:
        audio_input: The file path to the recorded audio from the gr.Audio input.
        history_state: The gr.State object holding the conversation history.

    Returns:
        A tuple containing:
        - The updated gr.State for conversation history.
        - The chatbot's audio response for the gr.Audio output.
        - The updated gr.Chatbot display.
    """
    if audio_input is None:
        return history_state, None, history_state or []

    logging.info(f"Received audio file: {audio_input}")

    # If history_state is None or not a list, initialize it.
    conversation_history = history_state if isinstance(history_state, list) else []

    # Process the audio and get the response
    response_text, audio_response, updated_history = asyncio.run(
        process_audio_input(audio_input, conversation_history)
    )

    # Format for Gradio chatbot display
    chatbot_display = []
    for i in range(0, len(updated_history), 2):
        user_turn = updated_history[i]["content"] if i < len(updated_history) else ""
        bot_turn = updated_history[i+1]["content"] if (i+1) < len(updated_history) else ""
        chatbot_display.append((user_turn, bot_turn))

    # The audio output requires a (sample_rate, numpy_array) tuple or a file path.
    # Since we have bytes, we save it to a temporary file for Gradio to play.
    # Note: A more direct way might be available in newer Gradio versions.
    audio_response_path = "response.wav"
    with open(audio_response_path, "wb") as f:
        f.write(audio_response)

    return updated_history, audio_response_path, chatbot_display

# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="Omani Therapist Voice Chat") as app:
    gr.Markdown(
        """
        # 🇴🇲 Omani Therapist Voice Chat
        أهلاً بك. اضغط على "Record from microphone" وتحدث باللغة العربية (اللهجة العمانية).
        """
    )

    # State to store the conversation history
    st_conversation_history = gr.State([])

    # UI Components
    chatbot_display = gr.Chatbot(label="Conversation", height=400)
    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak here")
    audio_output = gr.Audio(type="filepath", label="Bot Response", autoplay=True)

    # Connect the components
    audio_input.change(
        fn=voice_chat,
        inputs=[audio_input, st_conversation_history],
        outputs=[st_conversation_history, audio_output, chatbot_display],
        show_progress="full",
    )

if __name__ == "__main__":
    logging.info("Launching Gradio application...")
    app.launch(share=False) # Set share=True to get a public link
