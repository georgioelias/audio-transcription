import streamlit as st
from openai import OpenAI
from google.cloud import speech, storage
from google.oauth2 import service_account
from pydub import AudioSegment
from io import BytesIO
import os
import requests

# Set up the OpenAI client with the API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
gcreds = st.secrets["gcp_service_account"]
credentials = service_account.Credentials.from_service_account_info(gcreds)
speech_client = speech.SpeechClient(credentials=credentials)
storage_client = storage.Client(credentials=credentials)

st.title("Audio Transcription App")

# Service selection
transcription_service = st.radio(
    "Choose Transcription Service:",
    ("OpenAI Whisper", "Google Cloud Speech-to-Text", "Lemonfox Whisperv3")
)

if transcription_service == "OpenAI Whisper":
    # Whisper Transcription UI
    st.title("Whisper Transcription")

    # Sidebar for advanced settings
    st.sidebar.header("Advanced Settings")

    # Model Selection (only "whisper-1" available)
    model = "whisper-1"
    st.sidebar.write("**Model**: whisper-1")

    # Temperature
    temperature = st.sidebar.slider(
        "Temperature (controls randomness)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    # Format
    output_format = st.sidebar.selectbox("Output Format", ["text", "json"])

    # Timestamps (for json output only)
    timestamp_granularity = st.sidebar.selectbox(
        "Timestamp Granularity (only for JSON format)",
        ["none", "segment", "word", "both"],
        index=0
    ) if output_format == "json" else "none"

    # Function to limit the prompt to 224 tokens (approx. 1,500 characters)
    def limit_prompt(prompt, max_tokens=224):
        return prompt[:1500] if len(prompt) > 1500 else prompt

    # Initialize session state for storing the prompt
    if "saved_prompt" not in st.session_state:
        st.session_state.saved_prompt = ""

    # Prompt input in the main section
    prompt_text = st.text_area("Enter a prompt to guide the transcription (optional, max 224 tokens):")
    submit_prompt = st.button("Submit Prompt", key="submit_prompt")

    # Save the prompt when "Submit Prompt" is clicked
    if submit_prompt:
        st.session_state.saved_prompt = limit_prompt(prompt_text)
        st.success("Prompt saved successfully!")

elif transcription_service == "Google Cloud Speech-to-Text":
    # Google Cloud Speech-to-Text UI
    st.title("Google Cloud Speech-to-Text")
    
    # Sidebar settings for Google Cloud
    st.sidebar.header("Settings")
    output_format = st.sidebar.selectbox("Output Format", ["text", "json"])
    
    # Language selection
    language_options = {
        "English (US)": "en-US",
        "Spanish (Spain)": "es-ES",
        "French (France)": "fr-FR",
        "German (Germany)": "de-DE",
        "Chinese (Mandarin)": "zh",
        "Arabic (Saudi)": "ar-SA",
    }
    language_code = st.sidebar.selectbox(
        "Choose Language:",
        options=list(language_options.keys()),
        index=0  # Default to English (US)
    )
    selected_language_code = language_options[language_code]

    # Google Cloud Storage bucket name
    bucket_name = "your-bucket-name"

elif transcription_service == "Lemonfox Whisperv3":
    # Lemonfox Whisperv3 Transcription UI
    st.title("Lemonfox Whisperv3 Transcription")
    
    # Sidebar settings for Lemonfox Whisperv3
    st.sidebar.header("Lemonfox Settings")
    output_format = st.sidebar.selectbox("Output Format", ["text", "json"])
    
    # Language selection
    language_options = {
        "Auto-Detect" : " ",
        "English": "english",
        "Spanish": "spanish",
        "French": "french",
        "Arabic": "arabic",
    }
    language_code = st.sidebar.selectbox(
        "Choose Language:",
        options=list(language_options.keys()),
        index=0
    )
    selected_language_code = language_options[language_code]

    # Prompt input and submit button for Lemonfox
    if "lemonfox_saved_prompt" not in st.session_state:
        st.session_state.lemonfox_saved_prompt = ""
    
    lemonfox_prompt = st.text_area("Enter a prompt to guide the Lemonfox transcription (optional):")
    submit_lemonfox_prompt = st.button("Submit Prompt")

    # Save the prompt when "Submit Lemonfox Prompt" is clicked
    if submit_lemonfox_prompt:
        st.session_state.lemonfox_saved_prompt = lemonfox_prompt
        st.success("Prompt saved successfully!")

# Common audio input section
option = st.selectbox("Choose an option:", ("Record Audio", "Upload Audio"))
audio_file = None

if option == "Record Audio":
    try:
        audio_file = st.experimental_audio_input("Record your audio")
    except AttributeError:
        st.warning("Your Streamlit version doesn't support audio recording. Please upload an audio file instead.")
elif option == "Upload Audio":
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg", "opus"])

if audio_file:
    st.audio(audio_file, format="audio/wav")

def convert_to_wav(uploaded_file):
    audio_bytes = uploaded_file.read()
    audio_format = uploaded_file.name.split(".")[-1].lower()
    
    if audio_format in ["opus", "ogg"]:
        audio = AudioSegment.from_file(BytesIO(audio_bytes), format=audio_format)
    else:
        audio = AudioSegment.from_file(BytesIO(audio_bytes))
    
    wav_io = BytesIO()
    audio.set_frame_rate(16000).set_channels(1).export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

def upload_to_gcs(bucket_name, blob_name, file_data):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(file_data, rewind=True)
    return f"gs://{bucket_name}/{blob_name}"

# Transcribe button with unique key
transcribe_button = st.button("Transcribe", key="transcribe_button")

if transcribe_button and audio_file:
    try:
        if transcription_service == "OpenAI Whisper":
            # OpenAI Whisper Transcription logic
            audio_bytes = audio_file.read()
            audio_data = BytesIO(audio_bytes)
            audio_data.name = audio_file.name

            timestamp_options = {
                "none": None,
                "segment": ["segment"],
                "word": ["word"],
                "both": ["segment", "word"]
            }
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_data,
                prompt=st.session_state.saved_prompt or "",
                response_format="verbose_json" if output_format == "json" else "text",
                temperature=temperature,
                timestamp_granularities=timestamp_options.get(timestamp_granularity)
            )

            st.success("Transcription completed!")
            if output_format == "text":
                st.write("Transcription:")
                st.write(transcription)
            elif output_format == "json":
                st.json(transcription)

        elif transcription_service == "Google Cloud Speech-to-Text":
            # Google Cloud Speech-to-Text Transcription logic
            wav_file = convert_to_wav(audio_file)
            audio_size_mb = len(wav_file.getbuffer()) / (1024 * 1024)

            if audio_size_mb > 10:
                uri = upload_to_gcs(bucket_name, audio_file.name, wav_file)
                audio = speech.RecognitionAudio(uri=uri)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code=selected_language_code,
                )
                operation = speech_client.long_running_recognize(config=config, audio=audio)
                response = operation.result(timeout=300)
            else:
                audio = speech.RecognitionAudio(content=wav_file.read())
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code=selected_language_code,
                )
                response = speech_client.recognize(config=config, audio=audio)

            st.success("Transcription completed!")
            if output_format == "text":
                st.write("Transcription:")
                for result in response.results:
                    st.write(result.alternatives[0].transcript)
            elif output_format == "json":
                st.json([{
                    "transcript": result.alternatives[0].transcript,
                    "confidence": result.alternatives[0].confidence
                } for result in response.results])

        elif transcription_service == "Lemonfox Whisperv3":
            # Lemonfox Whisperv3 Transcription logic
            wav_file = convert_to_wav(audio_file)
            files = {"file": wav_file}
            data = {
                "language": selected_language_code,
                "response_format": output_format,
                "prompt": st.session_state.lemonfox_saved_prompt  # Use the saved prompt
            }
            headers = {
                "Authorization": st.secrets["LEMONFOX_API_KEY"]
            }
            response = requests.post("https://api.lemonfox.ai/v1/audio/transcriptions", headers=headers, files=files, data=data)
            
            if response.status_code == 200:
                try:
                    transcription_result = response.json()
                except ValueError:
                    transcription_result = response.text

                st.success("Transcription completed!")
                if output_format == "text":
                    st.write("Transcription:")
                    st.write(transcription_result.get("transcription", transcription_result) if isinstance(transcription_result, dict) else transcription_result)
                elif output_format == "json":
                    st.json(transcription_result)
            else:
                st.error(f"Transcription failed: {response.text}")

    except Exception as e:
        st.error(f"Error in transcription: {e}")

elif transcribe_button and not audio_file:
    st.warning("Please upload or record an audio file to transcribe.")
