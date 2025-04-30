import requests
import base64
import io
import wave
import json
import time
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import numpy as np # Needed for audio data conversion
from pydub import AudioSegment # Import pydub

# --- Configuration ---
# Updated endpoint based on documentation
JUVOLY_API_ENDPOINT = "https://services.juvoly.nl/api/v2/rest/speech/transcript?model=juvoly_v3"
CLIENT_ID = "uwv"
API_KEY = "462e0408-09d7-433d-a6b1-11479b9768a2"
# Changed back to Common Voice
DATASET_NAME = "mozilla-foundation/common_voice_13_0"
DATASET_LANG = "nl"
# Use the full test split name for streaming
DATASET_SPLIT = "test"
# Delay between API requests to avoid rate limiting (adjust if needed)
REQUEST_DELAY_SECONDS = 0.5
# Define how many samples to take from the stream
STREAM_TAKE_N = 10
# MAX_SAMPLES variable is no longer needed as we limit during loading
# MAX_SAMPLES = 10

# --- Helper Function: Convert NumPy array to WAV bytes ---
def numpy_to_wav_bytes(audio_array, sampling_rate):
    # Ensure array is in int16 format, typical for WAV
    if audio_array.dtype != np.int16:
        # Scale float audio to int16 range if necessary
        if np.issubdtype(audio_array.dtype, np.floating):
            # Normalize float array between -1 and 1 before scaling
            if np.max(np.abs(audio_array)) > 1.0:
                 audio_array = audio_array / np.max(np.abs(audio_array))
            audio_array = (audio_array * 32767).astype(np.int16)
        else:
             # Attempt conversion for other integer types if needed
             # This might require more specific handling depending on the source dtype
             audio_array = audio_array.astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)  # Assuming mono audio from Common Voice
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sampling_rate)
        wf.writeframes(audio_array.tobytes())
    return buffer.getvalue()

# --- Main Logic ---
# 1. Load Dataset
print(f"Loading dataset: {DATASET_NAME} ({DATASET_LANG}, split={DATASET_SPLIT}) in streaming mode...")
try:
    dataset = load_dataset(
        DATASET_NAME, 
        DATASET_LANG, 
        split=DATASET_SPLIT, 
        trust_remote_code=True, 
        token="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",
        streaming=True
    )
    # Limit the stream after loading
    if STREAM_TAKE_N is not None:
        print(f"Limiting stream to {STREAM_TAKE_N} samples.")
        dataset = dataset.take(STREAM_TAKE_N)
    print("Dataset loaded (streaming mode).")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have internet connectivity and the 'datasets' library is installed ('pip install datasets').")
    exit()


# 2. Load evaluation metric
print("Loading WER metric...")
try:
    wer = evaluate.load("wer")
    print("WER metric loaded.")
except Exception as e:
    print(f"Error loading WER metric: {e}")
    print("Please ensure the 'evaluate' and 'jiwer' libraries are installed ('pip install evaluate jiwer').")
    exit()


# 3. Transcription using Juvoly API and evaluation
predictions = []
references = []
headers = {
    # Corrected header names based on documentation
    "X-Juvoly-Client-Id": CLIENT_ID,
    "X-Juvoly-Api-Key": API_KEY,
    # Changed Content-Type to audio/mpeg for MP3
    "Content-Type": "audio/mpeg" 
}

print(f"Starting transcription using Juvoly API: {JUVOLY_API_ENDPOINT}")
# Removed note about assumed endpoint
# print("Note: Using an assumed batch endpoint. If transcription fails, please verify the correct HTTP endpoint in the script.")

for i, item in enumerate(tqdm(dataset, desc="Transcribing with Juvoly")):
    try:
        audio = item["audio"]
        # Changed key back to "sentence" for Common Voice dataset
        reference_text = item["sentence"] # Extract reference early

        # Basic check for valid audio data
        if not isinstance(audio, dict) or "array" not in audio or "sampling_rate" not in audio:
            print(f"Warning: Skipping item {i} due to unexpected audio format: {audio}")
            predictions.append("") # Keep lists aligned
            references.append(reference_text)
            continue

        audio_array = audio["array"]
        sampling_rate = audio["sampling_rate"]

        # Ensure audio_array is a numpy array
        if not isinstance(audio_array, np.ndarray):
             audio_array = np.array(audio_array)

        # Skip empty audio arrays
        if audio_array.size == 0:
            print(f"Warning: Skipping item {i} due to empty audio array.")
            predictions.append("") # Keep lists aligned
            references.append(reference_text)
            continue

        # Convert audio to WAV bytes
        wav_bytes = numpy_to_wav_bytes(audio_array, sampling_rate)

        # --- Convert WAV bytes to MP3 bytes using pydub ---
        try:
            wav_stream = io.BytesIO(wav_bytes)
            audio_segment = AudioSegment.from_wav(wav_stream)
            mp3_stream = io.BytesIO()
            # Export as MP3 using default bitrate
            audio_segment.export(mp3_stream, format="mp3") 
            mp3_bytes = mp3_stream.getvalue()
        except Exception as e:
            print(f"\nError converting audio to MP3 for item {i}: {e}")
            print("Ensure ffmpeg is installed and accessible in your system PATH.")
            # Skip this item if conversion fails
            predictions.append("")
            references.append(reference_text) 
            continue
        # --- End MP3 Conversion ---

        # Send request to Juvoly with raw MP3 bytes in the body
        response = requests.post(JUVOLY_API_ENDPOINT, headers=headers, data=mp3_bytes)

        # Print status/error for debugging
        if response.status_code != 200:
            print(f"\nWarning: API request for item {i} failed with status {response.status_code}. Response: {response.text}")
            response.raise_for_status() # Raise HTTPError to be caught below

        # Parse response based on documentation ("utterances" list)
        result = response.json()
        if "utterances" in result and len(result["utterances"]) > 0:
            # Assuming the first utterance contains the full transcript for short files
            # Concatenate sentences if multiple utterances are returned unexpectedly
            prediction_text = " ".join([utt.get("sentence", "") for utt in result["utterances"]]).strip()
            if not prediction_text:
                 print(f"\nWarning: Empty 'sentence' found in utterance(s) for item {i}. Response: {result}")
        else:
            print(f"\nWarning: 'utterances' key not found or empty in API response for item {i}. Response: {result}")
            prediction_text = "" # Or handle as error

        predictions.append(prediction_text)
        references.append(reference_text)

        # Add a delay
        time.sleep(REQUEST_DELAY_SECONDS)

    except requests.exceptions.RequestException as e:
        print(f"Error calling Juvoly API for item {i}: {e}")
        # Append an empty prediction and continue
        predictions.append("")
        references.append(reference_text) # Keep reference aligned
        print("Skipping item and continuing...")
        time.sleep(REQUEST_DELAY_SECONDS) # Still delay after error
    except KeyError as e:
         # Updated the key name in the error message if applicable
         print(f"Error processing dataset item {i}: Missing key {e}. Item structure: {item.keys()}. Expected 'audio' and 'sentence'.")
         # Skip this item as it's malformed or missing expected data
         predictions.append("")
         # Attempt to add reference if it was extracted, otherwise add empty string
         references.append(reference_text if 'reference_text' in locals() and i == len(references) else "")
         print("Skipping malformed item...")
    except Exception as e:
        print(f"An unexpected error occurred processing item {i}: {e}")
        # Append empty prediction and continue
        predictions.append("")
        references.append(reference_text if 'reference_text' in locals() and i == len(references) else "")
        print("Skipping item and continuing...")
        time.sleep(REQUEST_DELAY_SECONDS)


# 4. Compute Word Error Rate
print("Transcription complete. Calculating WER...")
if not predictions or not references:
     print("Error: Predictions or references list is empty. Cannot calculate WER.")
elif len(predictions) != len(references):
     print(f"Error: Mismatch between predictions ({len(predictions)}) and references ({len(references)}) count. Cannot calculate WER.")
else:
    try:
        wer_score = wer.compute(predictions=predictions, references=references)
        print(f"Word Error Rate (WER): {wer_score:.4f}")
    except Exception as e:
        print(f"Error calculating WER: {e}")

print(f"Processed {len(predictions)} items.")
print("Script finished.") 