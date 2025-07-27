import os
import shutil
import subprocess
import torch
import numpy as np
import json
from scipy.io.wavfile import write as write_wav
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import time
from mutagen.id3 import ID3, SYLT, Encoding
from multiprocessing import Process, Queue, set_start_method
import queue
from threading import Thread
import argparse
import re

# --- TTS Library Imports ---
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Register custom classes for PyTorch deserialization (needed for some TTS models)
torch.serialization.add_safe_globals([
    XttsConfig, Xtts, XttsAudioConfig, BaseDatasetConfig, XttsArgs
])

# --- Global variables for the worker process (used for multiprocessing) ---
tts_model_global = None
gpt_cond_latent_global = None
speaker_embedding_global = None


# =========================
# === HELPER FUNCTIONS  ===
# =========================

def update_progress_file(progress_path, chapter_num, status):
    """
    Updates the progress JSON file for a specific chapter.
    """
    with open(progress_path, 'r', encoding='utf-8') as f:
        progress_data = json.load(f)
    for chapter in progress_data["chapters"]:
        if chapter["num"] == chapter_num:
            chapter["status"] = status
            break
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=4)

def load_or_create_progress_file(progress_path, chapters_data, book_title):
    """Loads a progress file if it exists, otherwise creates a new one."""
    if os.path.exists(progress_path):
        print("Found existing progress file. Loading state.")
        with open(progress_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("No progress file found. Creating a new one.")
        progress_data = {
            "book_title": book_title,
            "chapters": [
                {"num": c["num"], "title": c["title"], "status": "pending"} for c in chapters_data
            ]
        }
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=4)
        return progress_data

def format_lrc_timestamp(seconds):
    """
    Converts a time in seconds to LRC timestamp format [mm:ss.xx].
    Used for synchronizing lyrics with audio.
    """
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    hundredths = int((seconds - (minutes * 60) - sec) * 100)
    return f"[{minutes:02d}:{sec:02d}.{hundredths:02d}]"

def extract_chapters_from_epub(epub_path, skip_start, skip_end):
    """
    Reads an EPUB file and extracts chapters as text.
    Skips a specified number of chapters at the start and end (e.g., for prefaces or appendices).
    Returns a list of dictionaries, each with chapter number, title, and text.
    """
    print(f"Reading EPUB file: {epub_path}")
    book = epub.read_epub(epub_path)  # Load the EPUB file
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))  # Get all document items (chapters, etc.)
    total_items = len(items)
    # Calculate how many chapters to skip at start and end
    start_skip = min(skip_start, total_items)
    end_skip = min(skip_end, total_items - start_skip)
    # Select only the chapters we want to process
    content_items = items[start_skip : total_items - end_skip]
    print(f"Found {len(content_items)} content chapters to process after skipping.")
    
    chapters_data = []
    for i, item in enumerate(content_items):
        # Parse the HTML content of the chapter
        soup = BeautifulSoup(item.get_body_content(), 'html.parser')
        
        # Try to find a chapter title in h1, h2, or h3 tags; fallback to "Chapter N"
        title_tag = soup.find('h1') or soup.find('h2') or soup.find('h3')
        title = title_tag.get_text(strip=True) if title_tag else f"Chapter {i + 1}"

        if title_tag:
            title_tag.decompose()  # Remove the title from the body so it doesn't appear twice

        # Try to get the main text from the <body> tag, fallback to all text
        body = soup.find('body')
        if body:
            text = body.get_text(separator='\n\n', strip=True)
        else:
            text = soup.get_text(separator='\n\n', strip=True)

        # Remove reference markers like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)

        # Only add chapters that have enough text (avoid empty or very short chapters)
        if text and len(text) > 100:
            chapters_data.append({"num": i + 1, "title": title, "text": text})
            
    return chapters_data

def robust_sentence_splitter(text, tts_synthesizer, max_len):
    """
    A more advanced sentence splitter that is paragraph-aware and also
    intelligently splits long sentences at natural punctuation boundaries.
    """
    print("Splitting text into intelligent, paragraph-aware chunks...")
    paragraphs = text.split('\n\n')
    final_chunks = []
    
    for para_index, para in enumerate(paragraphs):
        cleaned_para = para.replace('\n', ' ').strip()
        if not cleaned_para: continue
        
        sentences_from_para = tts_synthesizer.split_into_sentences(cleaned_para)
        
        for sent_index, sentence in enumerate(sentences_from_para):
            sanitized_sentence = sentence.strip(" \"'")
            if not sanitized_sentence or not any(c.isalnum() for c in sanitized_sentence): continue

            is_paragraph_end = (sent_index == len(sentences_from_para) - 1)

            # --- NEW, SMARTER SPLITTING LOGIC ---
            if len(sanitized_sentence) <= max_len:
                final_chunks.append({"text": sanitized_sentence, "is_para_end": is_paragraph_end})
            else:
                # The sentence is too long. We must split it intelligently.
                current_sentence_part = sanitized_sentence
                while len(current_sentence_part) > max_len:
                    # Find the last natural break point (punctuation) before the max length
                    split_pos = -1
                    for delimiter in ['.', ',', ';', ':', '—']:
                        pos = current_sentence_part.rfind(delimiter, 0, max_len)
                        if pos > split_pos:
                            split_pos = pos
                    
                    if split_pos == -1:
                        # No natural break found, fall back to the last space
                        split_pos = current_sentence_part.rfind(' ', 0, max_len)
                    
                    if split_pos == -1:
                        # No spaces found either, hard-cut the sentence (very rare)
                        split_pos = max_len - 1

                    # Add the first part of the split sentence
                    final_chunks.append({"text": current_sentence_part[:split_pos+1].strip(), "is_para_end": False})
                    # The remainder becomes the new part to process
                    current_sentence_part = current_sentence_part[split_pos+1:].strip()
                
                # Add the final remaining part of the sentence
                if current_sentence_part:
                    final_chunks.append({"text": current_sentence_part, "is_para_end": is_paragraph_end})

    return final_chunks


# =====================================
# === TTS WORKER PROCESS FUNCTION   ===
# =====================================

def tts_consumer(job_queue, results_queue, args):
    """
    This function runs in a separate process (worker).
    - Loads the TTS model and speaker embedding ONCE.
    - Waits for jobs (sentences to synthesize) from the job_queue.
    - Synthesizes audio for each sentence and adds a pause at the end.
    - Writes the audio to a WAV file and puts the result in results_queue.
    - Exits cleanly when it receives a "STOP" signal.
    """
    global tts_model_global, gpt_cond_latent_global, speaker_embedding_global

    # Only load the TTS model and speaker embedding once per process
    if tts_model_global is None:
        print("    [Worker Process] Initializing TTS model (this happens only once)...")
        tts_model_global = TTS(args.tts_model_name).to(args.device)
        print("    [Worker Process] Computing speaker latents (this happens only once)...")
        gpt_cond_latent_global, speaker_embedding_global = tts_model_global.synthesizer.tts_model.get_conditioning_latents(audio_path=args.voice_file)
        print("    [Worker Process] Worker is ready and waiting for jobs.")
    
    sample_rate = tts_model_global.synthesizer.output_sample_rate  # Get the output sample rate for audio

    while True:
        try:
            job = job_queue.get(timeout=5)  # Wait for a job from the queue
            if job == "STOP":
                print("\n    [Worker Process] STOP signal received. Exiting cleanly.")
                break  # Exit the loop and process

            # Unpack the job tuple
            idx, sentence_text, output_wav_path, pause_duration = job
            
            print(f"\r  > [Worker] Generating chunk {idx+1}...", end="", flush=True)

            # Generate audio for the sentence using the TTS model
            wav_chunk = tts_model_global.synthesizer.tts_model.inference(
                text=sentence_text, language="en",
                gpt_cond_latent=gpt_cond_latent_global, 
                speaker_embedding=speaker_embedding_global,
                enable_text_splitting=False, temperature=args.temperature, top_p=args.top_p
            )
            
            wav_data = wav_chunk['wav']  # Get the raw audio data (float32, -1.0 to 1.0)
            
            # If the model output is invalid (NaN or inf), replace with silence
            if not np.all(np.isfinite(wav_data)):
                print(f"\n--- WARNING: NaN/infinite values detected in audio for chunk {idx+1}. Skipping.")
                wav_data = np.zeros(1)

            # Convert float audio to int16 for WAV format
            int16_wav_data = np.int16(np.clip(wav_data, -1.0, 1.0) * 32767)

            # --- FIX: The worker now ONLY saves the raw speech ---
            write_wav(output_wav_path, sample_rate, int16_wav_data)
            
            # --- FIX: It now reports the duration of ONLY the speech ---
            duration_sec = len(int16_wav_data) / sample_rate
            results_queue.put((idx, sentence_text, duration_sec, output_wav_path))

        except queue.Empty:
            continue  # No job received, keep waiting
        except Exception as e:
            print(f"\n--- FATAL ERROR in TTS Consumer Process ---: {e}")
            break  # Exit on fatal error


# ===============================
# === MAIN ORCHESTRATOR LOGIC ===
# ===============================

def main(args):
    """
    Main function that coordinates the entire audiobook creation process.
    This is the definitive version with a persistent worker and a robust JSON checkpoint system.
    """
    start_time = time.time()
    
    # --- 1. SETUP ---
    book_output_dir = os.path.join(os.getcwd(), args.book_title)
    audio_chapters_dir = os.path.join(book_output_dir, "audio_chapters")
    os.makedirs(audio_chapters_dir, exist_ok=True)

    # Define the path for our central checkpoint file
    progress_file_path = os.path.join(book_output_dir, "generation_progress.json")

    # Step A: Parse the EPUB once to get the definitive list of all chapters
    chapters_from_epub = extract_chapters_from_epub(args.epub_file, args.skip_start, args.skip_end)
    if not chapters_from_epub:
        print("No chapters found in EPUB. Exiting.")
        return

    # Step B: If --force_reprocess is used, delete the old progress file
    if args.force_reprocess and os.path.exists(progress_file_path):
        print("--force_reprocess flag detected. Deleting old progress file.")
        os.remove(progress_file_path)
        
    # Step C: Load the progress file, or create a new one. This is our single source of truth.
    progress_data = load_or_create_progress_file(progress_file_path, chapters_from_epub, args.book_title)

    # Initialize the lightweight splitter and persistent worker process
    print("Initializing lightweight text splitter...")
    splitter_tts = TTS(args.tts_model_name)
    print("Splitter ready.")
    
    print("\n--- Starting Persistent Worker Process ---")
    job_queue = Queue(maxsize=64)
    results_queue = Queue(maxsize=64)
    consumer_process = Process(target=tts_consumer, args=(job_queue, results_queue, args))
    consumer_process.start()

    # --- MAIN GENERATION LOOP (NOW ITERATES OVER OUR PROGRESS FILE) ---
    for chapter_progress in progress_data["chapters"]:
        
        # This check is now based on the JSON status, not os.path.exists
        if chapter_progress["status"] == "complete":
            print(f"\n>>> Chapter {chapter_progress['num']}: '{chapter_progress['title']}' is already marked as complete. Skipping.")
            continue

        # Find the full chapter details from the list we loaded at the start
        chapter_info = next((c for c in chapters_from_epub if c["num"] == chapter_progress["num"]), None)
        if not chapter_info:
            print(f"Warning: Could not find chapter data for chapter number {chapter_progress['num']}. Skipping.")
            continue
        
        chapter_num = chapter_info["num"]
        chapter_title = chapter_info["title"]
        chapter_text = chapter_info["text"]
        chapter_audio_path = os.path.join(audio_chapters_dir, f"chapter_{chapter_num:04d}.mp3")

        print(f"\n>>> Processing Chapter {chapter_num}/{len(progress_data['chapters'])}: {chapter_title}")

        # Mark chapter as "in_progress" and save immediately
        chapter_progress["status"] = "in_progress"
        with open(progress_file_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=4)

        # It includes the temp folder creation, sentence splitting, collector thread,
        # producer loop, FFmpeg assembly, LRC creation, and Mutagen embedding.
        
        temp_chunk_folder = os.path.join(book_output_dir, "temp_audio_chunks")
        if os.path.exists(temp_chunk_folder): shutil.rmtree(temp_chunk_folder)
        os.makedirs(temp_chunk_folder)
        
        full_text_with_title = f"{chapter_title}\n\n{chapter_text}"
        sentence_chunks = robust_sentence_splitter(full_text_with_title, splitter_tts.synthesizer, args.max_len)
        total_chunks = len(sentence_chunks)

        chapter_timestamps = [None] * total_chunks
        sentence_files_ordered = [None] * total_chunks

        def collect_results():
            for _ in range(len(sentence_chunks)):
                try:
                    idx, text, duration, path = results_queue.get(timeout=args.collector_timeout)
                    chapter_timestamps[idx] = {"text": text, "duration": duration}
                    sentence_files_ordered[idx] = path
                except queue.Empty:
                    print(f"\n[Collector] CRITICAL: Timed out after {args.collector_timeout}s.")
                    break
        
        collector_thread = Thread(target=collect_results)
        collector_thread.start()

        for i, chunk_info in enumerate(sentence_chunks):
            sentence_text = chunk_info["text"]
            is_para_end = chunk_info["is_para_end"]
            pause_duration = args.para_pause if is_para_end else args.pause
            output_wav_path = os.path.join(temp_chunk_folder, f"s_{i:04d}.wav")
            job_queue.put((i, sentence_text, output_wav_path, pause_duration))
            print(f"\r  > [Producer] Sent job {i+1}/{total_chunks} to queue.", end="")

        print("\n  > [Producer] All jobs sent. Waiting for results...")
        collector_thread.join()
        
        print("\n  > Assembling and finalizing chapter MP3...")
        
        sample_rate = splitter_tts.synthesizer.output_sample_rate
        sent_pause_samples = int(args.pause * sample_rate)
        sent_silence_wav = np.zeros(sent_pause_samples, dtype=np.int16)
        sent_silence_path = os.path.join(temp_chunk_folder, "pause_sent.wav")
        write_wav(sent_silence_path, sample_rate, sent_silence_wav)
        
        para_pause_samples = int(args.para_pause * sample_rate)
        para_silence_wav = np.zeros(para_pause_samples, dtype=np.int16)
        para_silence_path = os.path.join(temp_chunk_folder, "pause_para.wav")
        write_wav(para_silence_path, sample_rate, para_silence_wav)

        filelist_path = os.path.join(temp_chunk_folder, "filelist.txt")
        with open(filelist_path, 'w', encoding='utf-8') as f:
            for i, (wav_file, chunk_info) in enumerate(zip(sentence_files_ordered, sentence_chunks)):
                if wav_file and os.path.exists(wav_file):
                    f.write(f"file '{os.path.basename(wav_file)}'\n")
                    if i < len(sentence_files_ordered) - 1:
                        if chunk_info["is_para_end"]:
                            f.write(f"file '{os.path.basename(para_silence_path)}'\n")
                        else:
                            f.write(f"file '{os.path.basename(sent_silence_path)}'\n")
        
        ffmpeg_command = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', os.path.basename(filelist_path),'-c', 'libmp3lame', '-b:a', '192k', os.path.abspath(chapter_audio_path)]
        try:
            subprocess.run(ffmpeg_command, check=True, capture_output=True, cwd=temp_chunk_folder)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] ffmpeg failed. Chapter marked to be re-processed.")
            chapter_progress["status"] = "pending"
            with open(progress_file_path, 'w', encoding='utf-8') as f: json.dump(progress_data, f, indent=4)
            continue

        chapter_lrc_path = chapter_audio_path.replace('.mp3', '.lrc')
        lrc_lines, current_time = [], 0.0
        for i, entry in enumerate(chapter_timestamps):
            if entry and entry['text'] and entry['duration'] > 0:
                lrc_lines.append(f"{format_lrc_timestamp(current_time)}{entry['text']}")
                current_time += entry['duration']
                if i < len(chapter_timestamps) - 1:
                    is_para_end = sentence_chunks[i]["is_para_end"]
                    current_time += args.para_pause if is_para_end else args.pause
        
        with open(chapter_lrc_path, 'w', encoding='utf-8') as f: f.write("\n".join(lrc_lines))
        
        try: audio = ID3(chapter_audio_path)
        except Exception: audio = ID3()
        with open(chapter_lrc_path, 'r', encoding='utf-8') as f: lrc_text = f.read()
        lrc_entries, lrc_pattern = [], re.compile(r"\[(\d+):(\d+)\.(\d+)\](.*)")
        for line in lrc_lines:
            match = lrc_pattern.match(line)
            if match:
                minutes, seconds, hundredths = int(match.group(1)), int(match.group(2)), int(match.group(3))
                timestamp_ms = (minutes * 60 + seconds) * 1000 + hundredths * 10
                lyric = match.group(4).strip()
                lrc_entries.append((lyric, timestamp_ms))
        
        sylt = SYLT(encoding=Encoding.UTF8, lang='eng', format=2, type=1, desc='Lyrics', text=lrc_entries)
        audio.setall('SYLT', [sylt])
        audio.save(chapter_audio_path, v2_version=3)
        
        shutil.rmtree(temp_chunk_folder)

        # After all steps for the chapter are successful, mark it as "complete"
        chapter_progress["status"] = "complete"
        with open(progress_file_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=4)
        
        print(f"--- Successfully completed and finalized Chapter {chapter_num} ---")
        # <<< END: FINAL CHECKPOINT UPDATE >>>

    # --- 5. SHUTDOWN AND FINISH ---
    print("\n--- All chapters processed! Shutting down worker process... ---")
    job_queue.put("STOP")
    consumer_process.join()
    
    print("\n--- Project Complete! ---")
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time/3600:.2f} hours.")
    print(f"Your chapterized audiobook is ready in: '{audio_chapters_dir}'")

# ===============================
# === ENTRY POINT / ARGPARSE  ===
# ===============================

if __name__ == "__main__":
    # Required for multiprocessing to work reliably on Windows and macOS
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(description="The Audiobook Factory: Generate a complete audiobook from an EPUB file.")
    
    # All of these arguments are required, you can change the default values to your own
    # OR you can use the commented-out versions to set default values
    parser.add_argument("--epub_file", type=str, required=True, help="Path to the input EPUB file.")
    # OR
   #parser.add_argument("-i", "--epub_file", type=str, default="./Your_Novel/ORV Vol. 1.epub", help="Path to the input EPUB file.")
     
    parser.add_argument("--voice_file", type=str, required=True, help="Path to the narrator's voice sample WAV file.")
    # OR
   #parser.add_argument("-v", "--voice_file", type=str, default="./narrator_voice/orv_narrator_voice.wav", help="Path to the narrator's voice sample WAV file.") 
   
    parser.add_argument("--book_title", type=str, required=True, help="The name of the book, used for the output folder.")
    #OR
   #parser.add_argument("-b", "--book_title", type=str, default="ORV Vol. 1", help="The name of the book, used for the output folder.") 
    parser.add_argument("--skip_start", type=int, default=6, help="Number of 'chapters' to skip at the beginning of the EPUB.")
    parser.add_argument("--skip_end", type=int, default=1, help="Number of 'chapters' to skip at the end of the EPUB.")
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds of silence to add between sentences.")
    parser.add_argument("--para_pause", type=float, default=1.2, help="A longer pause in seconds for paragraph breaks.")
    parser.add_argument("--max_len", type=int, default=240, help="Maximum character length for a single text chunk.")
    parser.add_argument("--temperature", type=float, default=0.8, help="TTS generation temperature.")
    parser.add_argument("--top_p", type=float, default=0.8, help="TTS generation top_p.")
    parser.add_argument("--collector_timeout", type=int, default=300, help="Seconds the collector will wait for a result.")
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing of all chapters, ignoring existing audio files.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for TTS ('cuda' or 'cpu').")
    parser.add_argument("--tts_model_name", type=str, default="tts_models/multilingual/multi-dataset/xtts_v2", help="The Coqui TTS model to use.")
    
    args = parser.parse_args()

    # --- Validate input files ---
    if not os.path.exists(args.epub_file):
        print(f"ERROR: EPUB file not found at '{args.epub_file}'")
        exit(1)
    if not os.path.exists(args.voice_file):
        print(f"ERROR: Narrator voice sample not found at '{args.voice_file}'")
        exit(1)
        
    # --- Start main orchestration ---
    main(args)

