import os
import torch
import numpy as np
from scipy.io.wavfile import write as write_wav
from TTS.api import TTS
import time

# These imports are good practice to prevent potential serialization errors on a fresh setup.
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# This manually tells PyTorch that these classes from the TTS library are safe to load.
torch.serialization.add_safe_globals([
    XttsConfig, 
    Xtts, 
    XttsAudioConfig, 
    BaseDatasetConfig,
    XttsArgs
])

# --- CONFIGURATION ---

# -- File Paths --
# A short text file (a few paragraphs) to test with.
INPUT_TEXT_FILE = "./Your_Novel/test.txt" 
# The same narrator voice you use for the main project.
NARRATOR_VOICE_SAMPLE = "./narrator_voice/narrator_voice.wav"
# The name of the output file. It will be overwritten on each run.
OUTPUT_AUDIO_FILE = "test_output.wav"

# --- PERFORMANCE TUNING DIALS ---
# This is where you experiment! Change these values and listen to the result.

# Suggestion: More Expressive (can be less stable) This is the best starting point.
# You can adjust these values to find the right balance between expressiveness and stability.
TEMPERATURE = 0.8  # Range: 0.0 (deterministic) to 1.0 (very random)
TOP_P = 0.8        # Range: 0.0 to 1.0. Helps prune low-probability words.

PAUSE_BETWEEN_SENTENCES = 0.5  # Add a small pause for realism
MAX_TEXT_LENGTH = 200          # Same as your main script


# --- Other advanced parameters (usually best to leave as default) ---
LENGTH_PENALTY = 1.0
REPETITION_PENALTY = 10.0
TOP_K = 50

# -----------------------------------------------------------------------------

def robust_sentence_splitter(text, tts_synthesizer, max_len):
    """A robust sentence splitter."""
    paragraphs = text.split('\n\n')
    all_sentences = []
    for para in paragraphs:
        cleaned_para = para.replace('\n', ' ').strip()
        if not cleaned_para: continue
        sentences_from_para = tts_synthesizer.split_into_sentences(cleaned_para)
        all_sentences.extend(sentences_from_para)
        
    final_chunks = []
    for sentence in all_sentences:
        sanitized_sentence = sentence.strip(" \"'")
        if not sanitized_sentence or not any(c.isalnum() for c in sanitized_sentence):
            continue
        if len(sanitized_sentence) <= max_len:
            final_chunks.append(sanitized_sentence)
        else: # Split overly long sentences
            current_sub_chunk = ""
            words = sanitized_sentence.split()
            for word in words:
                if len(current_sub_chunk) + len(word) + 1 > max_len:
                    if current_sub_chunk: final_chunks.append(current_sub_chunk.strip())
                    current_sub_chunk = word
                else: current_sub_chunk += " " + word
            if current_sub_chunk: final_chunks.append(current_sub_chunk.strip())
    return final_chunks

def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing Coqui XTTS model...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    print(f"Reading text from '{INPUT_TEXT_FILE}'...")
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = robust_sentence_splitter(text, tts.synthesizer, MAX_TEXT_LENGTH)
    
    print("\n--- Generating Test Audio ---")
    print(f" > Temperature: {TEMPERATURE}")
    print(f" > Top P: {TOP_P}")
    
    # --- OPTIMIZATION: Pre-compute the vocal fingerprint ONCE ---
    print("Computing speaker latents (vocal fingerprint)...")
    gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(audio_path=NARRATOR_VOICE_SAMPLE)
    print("... Done.")

    all_wav_data = []
    sample_rate = tts.synthesizer.output_sample_rate
    silence = np.zeros(int(PAUSE_BETWEEN_SENTENCES * sample_rate), dtype=np.float32)
    
    start_time = time.time()
    # Loop through sentences and use the fast, pre-computed fingerprint
    for i, sentence in enumerate(sentences):
        print(f"\r  > Processing test chunk {i+1}/{len(sentences)}...", end="")
        
        # Use the low-level inference function that we know works with pre-computed latents
        wav_chunk = tts.synthesizer.tts_model.inference(
            text=sentence,
            language="en",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            enable_text_splitting=False,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )
        wav_data = wav_chunk['wav']
        all_wav_data.append(wav_data)
        all_wav_data.append(silence)
        
    end_time = time.time()
    
    # Combine all audio chunks and silence into one long audio array
    final_wav_data = np.concatenate(all_wav_data)
    write_wav(OUTPUT_AUDIO_FILE, sample_rate, final_wav_data)

    total_time = end_time - start_time
    audio_duration = len(final_wav_data) / sample_rate
    
    print("\n\n--- Test Complete! ---")
    print(f" > Total Generation Time: {total_time:.2f} seconds")
    print(f" > Total Audio Duration: {audio_duration:.2f} seconds")
    print(f" > Real-Time Factor: {total_time / audio_duration:.2f}")
    print(f"\nListen to the result in '{OUTPUT_AUDIO_FILE}'")


if __name__ == "__main__":
    if not os.path.exists(NARRATOR_VOICE_SAMPLE):
        print(f"ERROR: Narrator voice sample not found at '{NARRATOR_VOICE_SAMPLE}'")
    elif not os.path.exists(INPUT_TEXT_FILE):
        print(f"ERROR: Please create a text file named '{INPUT_TEXT_FILE}' with some sample text.")
    else:
        run_test()