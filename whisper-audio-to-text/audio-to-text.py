import whisper

model = whisper.load_model("small.en") # base or medium

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an English .wav audio file to text using Whisper.

    Args:
        audio_path (str): Path to the .wav audio file

    Returns:
        str: Transcribed text from the audio
    """
    result = model.transcribe(audio_path, language="en")
    return result["text"]

text = transcribe_audio("DC_a01.wav")
print(text)
print(type(text))