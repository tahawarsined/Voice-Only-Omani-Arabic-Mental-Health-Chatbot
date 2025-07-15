import whisper
import ffmpeg
import os
import tempfile

# Load model once at import
WHISPER_MODEL = whisper.load_model("small")

def transcribe(filepath: str, lang="ar") -> str:
    """
    Transcribes an audio file, re-encoding via ffmpeg on failure.
    """
    try:
        # First attempt: direct transcription
        result = WHISPER_MODEL.transcribe(filepath, language=lang)
    except RuntimeError as decode_error:
        # On failure, re-encode to standard PCM WAV
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)  # Close the open handle

        try:
            # Re-encode to 16 kHz mono PCM WAV
            (
                ffmpeg
                .input(filepath)
                .output(temp_path,
                        format='wav',
                        acodec='pcm_s16le',
                        ac=1,
                        ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            # Retry transcription on the clean file
            result = WHISPER_MODEL.transcribe(temp_path, language=lang)
        finally:
            # Cleanup
            try:
                os.remove(temp_path)
            except OSError:
                pass

    # Return stripped text
    return result["text"].strip()
