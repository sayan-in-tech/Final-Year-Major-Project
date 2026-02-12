import os
import tempfile
import time
from pydub import AudioSegment

def convert_to_wav(uploaded_file, progress_callback=None):
    try:
        file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_input:
            temp_input.write(uploaded_file.read())
            input_path = temp_input.name

        if progress_callback:
            progress_callback(30, "📥 Reading and decoding audio...")

        # Load audio
        audio = AudioSegment.from_file(input_path)

        if progress_callback:
            progress_callback(60, "🔄 Converting to WAV format...")

        # Export to .wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            output_path = temp_wav.name
            audio.export(output_path, format="wav")

        if progress_callback:
            progress_callback(100, "✅ Conversion complete!")

        return output_path

    except Exception as e:
        print(f"[❌ ERROR] Audio conversion failed: {e}")
        if progress_callback:
            progress_callback(0, f"❌ Conversion failed: {e}")
        return None
import os
import tempfile
import time
from pydub import AudioSegment

def convert_to_wav(uploaded_file, progress_callback=None):
    try:
        file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_input:
            temp_input.write(uploaded_file.read())
            input_path = temp_input.name

        if progress_callback:
            progress_callback(30, "📥 Reading and decoding audio...")

        # Load audio
        audio = AudioSegment.from_file(input_path)

        if progress_callback:
            progress_callback(60, "🔄 Converting to WAV format...")

        # Export to .wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            output_path = temp_wav.name
            audio.export(output_path, format="wav")

        if progress_callback:
            progress_callback(100, "✅ Conversion complete!")

        return output_path

    except Exception as e:
        print(f"[❌ ERROR] Audio conversion failed: {e}")
        if progress_callback:
            progress_callback(0, f"❌ Conversion failed: {e}")
        return None
