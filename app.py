# ── app.py ────────────────────────────────────────────────────────────────────
"""
Gradio UI launcher for the Omani-Therapist-Voice chatbot.
"""

import asyncio
import logging
import gradio as gr
from src.pipeline import process_audio_input

logging.basicConfig(level=logging.INFO)


def voice_chat(audio_input, history_state):
    """
    Gradio callback: takes microphone audio + history_state → returns
    (new history_state, audio_response_path, chatbot_display_data)
    """
    # 0️⃣  Guard for first load / no recording yet
    if audio_input is None:
        return history_state or [], None, history_state or []

    logging.info(f"Received audio file: {audio_input}")

    # 1️⃣  Ensure history_state is a list
    conversation_history = history_state if isinstance(history_state, list) else []

    # 2️⃣  Run full pipeline
    response_text, audio_bytes, updated_history = asyncio.run(
        process_audio_input(audio_input, conversation_history)
    )

    # 3️⃣  Persist TTS bytes to a temp WAV for Gradio playback
    audio_response_path = None
    if audio_bytes:
        with open("response.wav", "wb") as fp:
            fp.write(audio_bytes)
        audio_response_path = "response.wav"

    # 4️⃣  Chatbot display → just feed the dict list the component wants
    chatbot_display = updated_history  # already [{'role':..., 'content':...}]

    # Return in the exact order specified in audio_input.change(...)
    return updated_history, audio_response_path, chatbot_display


# ── Gradio Interface ─────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="Omani Therapist Voice Chat") as app:
    gr.Markdown(
        """
        # 🇴🇲 Omani Therapist Voice Chat  
        أهلاً بك — اضغط “Record from microphone” وتحدث باللهجة العُمانية.
        """
    )

    st_conversation_history = gr.State([])

    chatbot_display = gr.Chatbot(
        label="Conversation",
        height=400,
        type="messages"   # expects list[{"role":..,"content":..}]
    )
    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Speak here"
    )
    audio_output = gr.Audio(
        type="filepath",
        label="Bot Response",
        autoplay=True
    )

    audio_input.change(
        fn=voice_chat,
        inputs=[audio_input, st_conversation_history],
        outputs=[st_conversation_history, audio_output, chatbot_display],
        show_progress="full",
    )

if __name__ == "__main__":
    logging.info("Launching Gradio application...")
    app.launch(share=False)   # set share=True for a public link
# ─────────────────────────────────────────────────────────────────────────────
