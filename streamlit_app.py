import streamlit as st
import requests
import io
import os # Added for environment variables
import time # Added for delay
from dotenv import load_dotenv # Added for .env
from pydub import AudioSegment
import pydub.exceptions
from datasets import load_dataset # Added for dataset testing
import evaluate # Added for dataset testing
import numpy as np # Added for dataset testing

# --- Load Environment Variables --- 
load_dotenv() 

# --- Configuration --- 
# Load secrets from environment variables
JUVOLY_API_ENDPOINT = "https://services.juvoly.nl/api/v2/rest/speech/transcript?model=juvoly_v3"
CLIENT_ID = os.getenv("JUVOLY_CLIENT_ID")
API_KEY = os.getenv("JUVOLY_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN") # Needed for Common Voice

# Dataset configuration (can be made configurable later)
DATASET_NAME = "mozilla-foundation/common_voice_13_0"
DATASET_LANG = "nl"
DATASET_SPLIT = "test" # Base split for streaming
REQUEST_DELAY_SECONDS = 0.5 # Delay between API requests in batch test

# --- Helper Function: Convert NumPy array to WAV bytes (from juvoly_test.py) ---
def numpy_to_wav_bytes(audio_array, sampling_rate):
    # Ensure array is in int16 format, typical for WAV
    if audio_array.dtype != np.int16:
        if np.issubdtype(audio_array.dtype, np.floating):
            if np.max(np.abs(audio_array)) > 1.0:
                 audio_array = audio_array / np.max(np.abs(audio_array))
            audio_array = (audio_array * 32767).astype(np.int16)
        else:
             audio_array = audio_array.astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)  # Assuming mono audio
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sampling_rate)
        wf.writeframes(audio_array.tobytes())
    return buffer.getvalue()

# --- Helper Function: Transcribe Audio (using loaded secrets) ---
def transcribe_audio(audio_bytes):
    """Sends audio bytes to Juvoly API and returns the transcription."""
    if not CLIENT_ID or not API_KEY:
        st.error("Juvoly Client ID of API Key niet gevonden in .env bestand.")
        return None
        
    headers = {
        "X-Juvoly-Client-Id": CLIENT_ID,
        "X-Juvoly-Api-Key": API_KEY,
        "Content-Type": "audio/mpeg" 
    }

    try:
        # Use BytesIO to handle the audio bytes in memory
        audio_stream = io.BytesIO(audio_bytes)
        
        # Load audio using pydub (supports various formats)
        try:
            audio_segment = AudioSegment.from_file(audio_stream)
        except pydub.exceptions.CouldntDecodeError:
             st.error("Kon het audiobestand niet decoderen. Zorg ervoor dat het een ondersteund formaat is (bv. WAV, MP3).")
             return None
        except Exception as e: # Catch other potential pydub errors
             st.error(f"Fout bij het laden van audio met pydub: {e}")
             return None

        # Export as MP3 in memory
        mp3_stream = io.BytesIO()
        try:
            audio_segment.export(mp3_stream, format="mp3") # Using default bitrate
            mp3_bytes = mp3_stream.getvalue()
        except Exception as e:
            st.error(f"Fout bij converteren naar MP3: {e}")
            st.warning("Zorg ervoor dat ffmpeg correct geïnstalleerd is.")
            return None

        # Send request to Juvoly
        response = requests.post(JUVOLY_API_ENDPOINT, headers=headers, data=mp3_bytes)

        # Handle response
        if response.status_code == 200:
            result = response.json()
            if "utterances" in result and len(result["utterances"]) > 0:
                transcription = " ".join([utt.get("sentence", "") for utt in result["utterances"]]).strip()
                if not transcription:
                    # Return empty string for empty transcription, don't show warning here
                    return ""
                return transcription
            else:
                # Return None for unexpected format, error handled later
                 return None
        else:
            # Return None on API error, handled later
            return None

    except requests.exceptions.RequestException as e:
        # Return None on network error, handled later
        return None
    except Exception as e:
        # Return None on unexpected error, handled later
        return None

# --- Helper Function: Run Dataset Test ---
def run_dataset_test(num_samples):
    """Loads dataset, runs transcription test, returns WER."""
    if not CLIENT_ID or not API_KEY:
        st.error("Juvoly Client ID of API Key niet gevonden in .env bestand.")
        return None, 0
    if not HUGGINGFACE_TOKEN:
        st.error("Hugging Face Token niet gevonden in .env bestand. Nodig voor Common Voice.")
        return None, 0

    predictions = []
    references = []
    processed_count = 0
    total_samples = num_samples

    try:
        # Load dataset in streaming mode
        st.info(f"Laden van dataset: {DATASET_NAME} ({DATASET_LANG}, split={DATASET_SPLIT}) (streaming)... Dit kan even duren bij eerste keer.")
        dataset = load_dataset(
            DATASET_NAME, 
            DATASET_LANG, 
            split=DATASET_SPLIT, 
            trust_remote_code=True, 
            token=HUGGINGFACE_TOKEN,
            streaming=True
        )
        
        # Limit the stream 
        if num_samples is not None:
            st.info(f"Beperken tot {num_samples} samples.")
            dataset = dataset.take(num_samples)
        else:
            st.warning("Volledige testset wordt verwerkt. Dit kan lang duren!")
            # Note: Getting total count for progress bar in streaming is tricky/slow.
            # We'll just show progress based on num_samples if provided.
            total_samples = None # Indicate unknown total for progress bar

        # Load WER metric
        wer_metric = evaluate.load("wer")
        
        st.info("Starten van transcripties via Juvoly API...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, item in enumerate(dataset):
            if total_samples: # Update progress bar only if we know the total
                 progress = (i + 1) / total_samples
                 progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Verwerken sample {i+1}...")

            try:
                audio = item["audio"]
                reference_text = item["sentence"]
                
                if not isinstance(audio, dict) or "array" not in audio or "sampling_rate" not in audio:
                    st.warning(f"Sample {i+1}: Ongeldig audio formaat, wordt overgeslagen.")
                    predictions.append("") # Keep lists aligned for WER
                    references.append(reference_text)
                    processed_count += 1
                    continue
                    
                audio_array = audio["array"]
                sampling_rate = audio["sampling_rate"]

                if not isinstance(audio_array, np.ndarray):
                     audio_array = np.array(audio_array)
                if audio_array.size == 0:
                    st.warning(f"Sample {i+1}: Lege audio array, wordt overgeslagen.")
                    predictions.append("")
                    references.append(reference_text)
                    processed_count += 1
                    continue

                # --- Get MP3 bytes (reuse transcribe_audio's conversion part, slightly adapted) ---
                wav_bytes = numpy_to_wav_bytes(audio_array, sampling_rate)
                wav_stream = io.BytesIO(wav_bytes)
                audio_segment = AudioSegment.from_wav(wav_stream)
                mp3_stream = io.BytesIO()
                audio_segment.export(mp3_stream, format="mp3")
                mp3_bytes = mp3_stream.getvalue()
                # --- End MP3 Conversion --- 
                
                # Call API (reuse transcription function without pydub conversion)
                headers = {"X-Juvoly-Client-Id": CLIENT_ID, "X-Juvoly-Api-Key": API_KEY, "Content-Type": "audio/mpeg"}
                response = requests.post(JUVOLY_API_ENDPOINT, headers=headers, data=mp3_bytes)
                
                prediction_text = ""
                if response.status_code == 200:
                    result = response.json()
                    if "utterances" in result and len(result["utterances"]) > 0:
                        prediction_text = " ".join([utt.get("sentence", "") for utt in result["utterances"]]).strip()
                    else:
                        st.warning(f"Sample {i+1}: Onverwachte response (geen utterances): {result}")
                else:
                    st.warning(f"Sample {i+1}: API Fout ({response.status_code}): {response.text}")
                
                predictions.append(prediction_text)
                references.append(reference_text)
                processed_count += 1
                
                time.sleep(REQUEST_DELAY_SECONDS) # Delay between requests

            except Exception as e:
                st.error(f"Fout bij verwerken sample {i+1}: {e}")
                # Add placeholders to keep lists aligned if an error occurs mid-sample processing
                if len(predictions) == i:
                     predictions.append("")
                if len(references) == i:
                     references.append(reference_text if 'reference_text' in locals() else "") # Try to add reference
                processed_count += 1 # Count as processed even if failed
                continue # Continue to next sample
        
        status_text.text("Transcriptie voltooid. Berekenen van WER...")

        if not predictions or not references:
            st.error("Geen resultaten om WER te berekenen.")
            return None, processed_count
        if len(predictions) != len(references):
             st.error(f"Fout: Mismatch tussen aantal voorspellingen ({len(predictions)}) en referenties ({len(references)}).")
             return None, processed_count
             
        wer_score = wer_metric.compute(predictions=predictions, references=references)
        progress_bar.progress(1.0)
        status_text.text("Dataset test voltooid.")
        return wer_score, processed_count

    except Exception as e:
        st.error(f"Fout tijdens dataset test: {e}")
        return None, processed_count

# --- Streamlit App UI --- 
st.set_page_config(page_title="Juvoly Test App", layout="wide")
st.title("🎙️ Juvoly Test Applicatie")

# Check if secrets are loaded
if not CLIENT_ID or not API_KEY:
    st.error("Kon Juvoly credentials niet laden uit `.env` bestand. Zorg dat het bestand bestaat en `JUVOLY_CLIENT_ID` en `JUVOLY_API_KEY` bevat.")
    st.stop()

tab1, tab2 = st.tabs(["📄 Enkel Bestand Transcriptie", "📊 Dataset Test (Common Voice)"])

# --- Tab 1: Single File Upload --- 
with tab1:
    st.header("Enkel Bestand Transcriptie")
    st.write("Upload een audiobestand (bv. WAV, MP3) om het te laten transcriberen door Juvoly V3.")
    uploaded_file = st.file_uploader("Kies een audiobestand", type=["wav", "mp3", "m4a", "ogg", "flac"], key="single_uploader")

    if uploaded_file is not None:
        st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[-1]}') 
        audio_bytes = uploaded_file.getvalue()

        if st.button("Start Transcriptie", key="single_transcribe_button"):
            with st.spinner("Audio wordt verwerkt door Juvoly..."):
                transcription_result = transcribe_audio(audio_bytes)
            
            if transcription_result is not None:
                st.subheader("Transcriptie Resultaat:")
                st.text_area("Herkende tekst", transcription_result, height=200, key="single_result")
            # Errors displayed within transcribe_audio or if it returns None
            elif transcription_result is None: 
                 st.error("Transcriptie mislukt. Zie eerdere foutmeldingen.")
    else:
        st.info("Wachtend op upload...")

# --- Tab 2: Dataset Test --- 
with tab2:
    st.header("Dataset Test (Common Voice - NL)")
    st.write("Test de Juvoly API op de Nederlandse Common Voice testset (via streaming). Let op: dit kan lang duren en veel API calls maken.")
    
    if not HUGGINGFACE_TOKEN:
         st.error("Hugging Face Token niet gevonden in `.env` bestand (`HUGGINGFACE_TOKEN=...`). Nodig om Common Voice te laden.")
    else:
        num_samples_to_test = st.number_input("Aantal samples om te testen", min_value=1, value=10, step=1, key="num_samples")
        
        if st.button("Start Dataset Test", key="dataset_test_button"):
            st.info(f"Starten van test met {num_samples_to_test} samples...")
            wer_score, processed_count = run_dataset_test(num_samples_to_test)
            
            st.subheader("Test Resultaten")
            st.metric(label="Aantal Verwerkte Samples", value=processed_count)
            if wer_score is not None:
                st.metric(label="Word Error Rate (WER)", value=f"{wer_score:.4f}")
                st.caption("(Lager is beter)")
            else:
                st.error("Kon WER niet berekenen vanwege een fout tijdens de test.") 