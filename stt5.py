import streamlit as st
from openai import OpenAI
from google.cloud import speech, storage
from google.oauth2 import service_account
from pydub import AudioSegment
from io import BytesIO
import os
import requests
import assemblyai as aai

# Set up the OpenAI client with the API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
gcreds = st.secrets["gcp_service_account"]
credentials = service_account.Credentials.from_service_account_info(gcreds)
speech_client = speech.SpeechClient(credentials=credentials)
storage_client = storage.Client(credentials=credentials)
aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]
lemonfox_apikey= st.secrets["LEMONFOX_API_KEY"]
st.title("Audio Transcription App")

# Service selection
transcription_service = st.radio(
    "Choose Transcription Service:",
    ("OpenAI Whisper", "Google Cloud Speech-to-Text", "Lemonfox Whisperv3", "AssemblyAI")
)

if transcription_service == "OpenAI Whisper":
    # Whisper Transcription UI
    st.title("Whisper Transcription")

    # Sidebar for advanced settings
    st.sidebar.header("Advanced Settings")
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
    output_format = st.sidebar.selectbox("Output Format", ["text", "json"])

    timestamp_granularity = st.sidebar.selectbox(
        "Timestamp Granularity (only for JSON format)",
        ["none", "segment", "word", "both"],
        index=0
    ) if output_format == "json" else "none"

    def limit_prompt(prompt, max_tokens=224):
        return prompt[:1500] if len(prompt) > 1500 else prompt

    if "saved_prompt" not in st.session_state:
        st.session_state.saved_prompt = ""

    prompt_text = st.text_area("Enter a prompt to guide the transcription (optional, max 224 tokens):")
    submit_prompt = st.button("Submit Prompt", key="submit_prompt")

    if submit_prompt:
        st.session_state.saved_prompt = limit_prompt(prompt_text)
        st.success("Prompt saved successfully!")

elif transcription_service == "Google Cloud Speech-to-Text":
    # Google Cloud Speech-to-Text UI
    st.title("Google Cloud Speech-to-Text")
    st.sidebar.header("Settings")
    output_format = st.sidebar.selectbox("Output Format", ["text", "json"])

    language_options = {
        "English (US)": "en-US",
        "Spanish (Spain)": "es-ES",
        "French (France)": "fr-FR",
        "German (Germany)": "de-DE",
        "Chinese (Mandarin)": "zh",
        "Arabic (Saudi)": "ar-SA",
    }
    language_code = st.sidebar.selectbox("Choose Language:", options=list(language_options.keys()))
    selected_language_code = language_options[language_code]
    bucket_name = "your-bucket-name"

elif transcription_service == "Lemonfox Whisperv3":
    # Lemonfox Whisperv3 Transcription UI
    st.title("Lemonfox Whisperv3 Transcription")
    st.sidebar.header("Lemonfox Settings")
    output_format = st.sidebar.selectbox("Output Format", ["text", "json"])

    language_options = {
        "Auto-Detect": " ",
        "English": "english",
        "Spanish": "spanish",
        "French": "french",
        "Arabic": "arabic",
    }
    language_code = st.sidebar.selectbox("Choose Language:", options=list(language_options.keys()))
    selected_language_code = language_options[language_code]

    if "lemonfox_saved_prompt" not in st.session_state:
        st.session_state.lemonfox_saved_prompt = ""
    
    lemonfox_prompt = st.text_area("Enter a prompt to guide the Lemonfox transcription (optional):")
    submit_lemonfox_prompt = st.button("Submit Prompt")

    if submit_lemonfox_prompt:
        st.session_state.lemonfox_saved_prompt = lemonfox_prompt
        st.success("Prompt saved successfully!")

elif transcription_service == "AssemblyAI":
    # AssemblyAI Transcription UI
    st.title("AssemblyAI Transcription")
    st.sidebar.header("AssemblyAI Settings")

    # Model Selection
    model_selection = st.sidebar.selectbox("Select Speech Model", ("Best (default)", "Nano"))
    speech_model = aai.SpeechModel.best if model_selection == "Best (default)" else aai.SpeechModel.nano

    # Punctuation and Casing
    enable_punctuation = st.sidebar.checkbox("Enable Automatic Punctuation and Casing", value=True)

    # Language Detection
    enable_language_detection = st.sidebar.checkbox("Enable Automatic Language Detection", value=False)
    language_confidence_threshold = None
    if enable_language_detection:
        language_confidence_threshold = st.sidebar.slider("Language Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

    # Specify Language Manually
    manual_language_code = st.sidebar.text_input("Specify Language Code (e.g., 'en')", value="")

    # Speaker Labels
    speaker_labels = st.sidebar.checkbox("Enable Speaker Labels", value=False)

    # Custom Spelling
    custom_spelling = st.sidebar.text_area("Custom Spelling (JSON format, e.g., {'Gettleman': ['gettleman'], 'SQL': ['Sequel']})")
    custom_spelling_dict = {}
    if custom_spelling:
        try:
            custom_spelling_dict = eval(custom_spelling)
        except Exception:
            st.sidebar.warning("Invalid format for custom spelling. Please use a JSON format.")

    # Custom Vocabulary
    custom_vocabulary = st.sidebar.text_area("Custom Vocabulary (comma-separated)")
    boost_level = st.sidebar.selectbox("Boost Level for Custom Vocabulary", ["default", "low", "high"])

    # Multichannel Transcription
    multichannel = st.sidebar.checkbox("Enable Multichannel Transcription", value=False)

    # Export Options
    export_srt = st.sidebar.checkbox("Export SRT Captions")
    export_vtt = st.sidebar.checkbox("Export VTT Captions")
    chars_per_caption = st.sidebar.slider("Characters Per Caption", min_value=20, max_value=100, value=32)

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
    audio = AudioSegment.from_file(BytesIO(audio_bytes), format=audio_format)
    wav_io = BytesIO()
    audio.set_frame_rate(16000).set_channels(1).export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

def upload_to_gcs(bucket_name, blob_name, file_data):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(file_data, rewind=True)
    return f"gs://{bucket_name}/{blob_name}"

transcribe_button = st.button("Transcribe", key="transcribe_button")

if transcribe_button and audio_file:
    try:
        if transcription_service == "OpenAI Whisper":
            audio_bytes = audio_file.read()
            audio_data = BytesIO(audio_bytes)
            audio_data.name = audio_file.name
            timestamp_options = {"none": None, "segment": ["segment"], "word": ["word"], "both": ["segment", "word"]}
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_data,
                prompt=st.session_state.saved_prompt or "",
                response_format="verbose_json" if output_format == "json" else "text",
                temperature=temperature,
                timestamp_granularities=timestamp_options.get(timestamp_granularity)
            )
            st.success("Transcription completed!")
            st.write(transcription if output_format == "text" else transcription)

        elif transcription_service == "Google Cloud Speech-to-Text":
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
            if response and output_format == "text":
                for result in response.results:
                    st.write(result.alternatives[0].transcript)
            elif response and output_format == "json":
                st.json([{
                    "transcript": result.alternatives[0].transcript,
                    "confidence": result.alternatives[0].confidence
                } for result in response.results])

        elif transcription_service == "Lemonfox Whisperv3":
            wav_file = convert_to_wav(audio_file)
            files = {"file": wav_file}
            data = {
                "language": selected_language_code,
                "response_format": output_format,
                "prompt": st.session_state.lemonfox_saved_prompt
            }
            headers = {
                "Authorization": lemonfox_apikey
            }
            response = requests.post("https://api.lemonfox.ai/v1/audio/transcriptions", headers=headers, files=files, data=data)
            
            if response.status_code == 200:
                try:
                    transcription_result = response.json()
                except ValueError:
                    transcription_result = response.text

                st.success("Transcription completed!")
                if output_format == "text":
                    st.write(transcription_result.get("transcription", transcription_result) if isinstance(transcription_result, dict) else transcription_result)
                elif output_format == "json":
                    st.json(transcription_result)
            else:
                st.error(f"Transcription failed: {response.text}")

        elif transcription_service == "AssemblyAI":
            wav_file = convert_to_wav(audio_file)
            audio_bytes = wav_file.read()
            transcriber = aai.Transcriber()
            
            # Configure AssemblyAI settings
            config = aai.TranscriptionConfig(
                speech_model=speech_model,
                punctuate=enable_punctuation,
                format_text=enable_punctuation,
                language_detection=enable_language_detection,
                language_confidence_threshold=language_confidence_threshold,
                language_code=manual_language_code if manual_language_code else None,
                speaker_labels=speaker_labels,
                multichannel=multichannel,
                word_boost=custom_vocabulary.split(",") if custom_vocabulary else None,
                boost_param=boost_level
            )
            
            if custom_spelling_dict:
                config.set_custom_spelling(custom_spelling_dict)

            transcript = transcriber.transcribe(audio_bytes, config)
            
            if transcript.status == aai.TranscriptStatus.error:
                st.error(f"Transcription failed: {transcript.error}")
            else:
                st.success("Transcription completed!")
                st.write(transcript.text)
                
                # Display utterances with speaker labels if enabled
                if speaker_labels:
                    for utterance in transcript.utterances:
                        st.write(f"Speaker {utterance.speaker}: {utterance.text}")
                
                # Export captions if selected
                if export_srt:
                    srt_captions = transcript.export_subtitles_srt(chars_per_caption=chars_per_caption)
                    st.download_button("Download SRT Captions", srt_captions, "captions.srt", "text/plain")

                if export_vtt:
                    vtt_captions = transcript.export_subtitles_vtt(chars_per_caption=chars_per_caption)
                    st.download_button("Download VTT Captions", vtt_captions, "captions.vtt", "text/vtt")

    except Exception as e:
        st.error(f"Error in transcription: {e}")

elif transcribe_button and not audio_file:
    st.warning("Please upload or record an audio file to transcribe.")
