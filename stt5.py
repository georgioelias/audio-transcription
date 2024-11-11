import streamlit as st
from openai import OpenAI
from google.cloud import speech, storage
from google.oauth2 import service_account
from pydub import AudioSegment
from io import BytesIO
import os

# Set up the OpenAI client with the API key
client = OpenAI(api_key= st.secrets["OPENAI_API_KEY"])
gcreds = st.secrets["gcp_service_account"]
# Initialize Google Cloud clients
credentials = service_account.Credentials.from_service_account_file(gcreds)
speech_client = speech.SpeechClient(credentials=credentials)
storage_client = storage.Client(credentials=credentials)

st.title("Audio Transcription App")

# Service selection
transcription_service = st.radio(
    "Choose Transcription Service:",
    ("OpenAI Whisper", "Google Cloud Speech-to-Text")
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

else:
    # Google Cloud Speech-to-Text UI
    st.title("Google Cloud Speech-to-Text")
    
    # Sidebar settings for Google Cloud
    st.sidebar.header("Settings")
    output_format = st.sidebar.selectbox("Output Format", ["text", "json"])
    
    # Google Cloud Storage bucket name
    bucket_name = "your-bucket-name"

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
            # Read the uploaded file and prepare it for transcription
            audio_bytes = audio_file.read()
            
            # Convert .ogg or .opus to .wav if necessary
            if audio_file.type in ["audio/ogg", "audio/opus"]:
                audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="ogg" if audio_file.type == "audio/ogg" else "opus")
                audio_data = BytesIO()
                audio_segment.export(audio_data, format="wav")
                audio_data.name = audio_file.name.replace(".ogg", ".wav").replace(".opus", ".wav")
            else:
                audio_data = BytesIO(audio_bytes)
                audio_data.name = audio_file.name

            # Define granularity for JSON output
            granularity_options = {
                "none": None,
                "segment": ["segment"],
                "word": ["word"],
                "both": ["segment", "word"]
            }
            timestamp_options = granularity_options[timestamp_granularity]

            # Transcribe audio using the client object
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_data,
                prompt=st.session_state.saved_prompt or "",
                response_format="verbose_json" if output_format == "json" else "text",
                temperature=temperature,
                timestamp_granularities=timestamp_options if timestamp_options else None
            )
            
            # Display the transcription result based on the output format
            if output_format == "text":
                st.write("Transcription:")
                st.write(transcription)
            elif output_format == "json":
                st.write("Transcription JSON:")
                st.json(transcription)

                # Display word-level timestamps if available
                if timestamp_granularity in ["word", "both"] and hasattr(transcription, "words"):
                    st.write("Word-Level Timestamps:")
                    for word_info in transcription.words:
                        st.write(word_info)

        else:
            # Google Cloud Speech-to-Text transcription
            wav_file = convert_to_wav(audio_file)
            audio_size_mb = len(wav_file.getbuffer()) / (1024 * 1024)

            if audio_size_mb > 10:
                st.info("Uploading file to Google Cloud Storage for asynchronous transcription...")
                uri = upload_to_gcs(bucket_name, audio_file.name, wav_file)
                st.success("File uploaded successfully!")

                # Run asynchronous transcription
                st.info("Transcribing asynchronously...")
                audio = speech.RecognitionAudio(uri=uri)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en-US",
                )
                operation = speech_client.long_running_recognize(config=config, audio=audio)
                response = operation.result(timeout=300)
            else:
                # Run synchronous transcription
                st.info("Transcribing synchronously...")
                audio = speech.RecognitionAudio(content=wav_file.read())
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en-US",
                )
                response = speech_client.recognize(config=config, audio=audio)

            # Display transcription
            st.success("Transcription completed!")
            if output_format == "text":
                st.write("Transcription:")
                st.write("\n".join(result.alternatives[0].transcript for result in response.results))
            elif output_format == "json":
                st.json([{
                    "transcript": result.alternatives[0].transcript,
                    "confidence": result.alternatives[0].confidence
                } for result in response.results])

    except Exception as e:
        st.error(f"Error in transcription: {e}")
elif transcribe_button and not audio_file:
    st.warning("Please upload or record an audio file to transcribe.")
