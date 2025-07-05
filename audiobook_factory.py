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

def load_or_create_progress_file(progress_path, chapters_data):
    """Loads a progress file if it exists, otherwise creates a new one."""
    if os.path.exists(progress_path):
        print("Found existing progress file. Loading state.")
        with open(progress_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("No progress file found. Creating a new one.")
        progress_data = {
            "book_title": chapters_data[0].get("book_title", "Unknown"),
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
    Splits a large block of text into smaller chunks for TTS.
    - Splits by paragraphs first, then by sentences.
    - If a sentence is too long, splits by words to keep each chunk <= max_len characters.
    - Marks the end of paragraphs for longer pauses.
    Returns a list of dicts: {text, is_para_end}
    """
    print("Splitting text into paragraph-aware chunks...")
    paragraphs = text.split('\n\n')  # Split by double newlines (paragraphs)
    final_chunks = []
    
    for para_index, para in enumerate(paragraphs):
        cleaned_para = para.replace('\n', ' ').strip()  # Remove single newlines and extra spaces
        if not cleaned_para:
            continue  # Skip empty paragraphs
        
        # Use TTS's built-in sentence splitter for better accuracy
        sentences_from_para = tts_synthesizer.split_into_sentences(cleaned_para)
        
        for sent_index, sentence in enumerate(sentences_from_para):
            sanitized_sentence = sentence.strip(" \"'")  # Remove leading/trailing quotes and spaces
            # Skip empty or non-alphanumeric sentences
            if not sanitized_sentence or not any(c.isalnum() for c in sanitized_sentence):
                continue

            is_paragraph_end = (sent_index == len(sentences_from_para) - 1)  # True if last sentence in paragraph

            if len(sanitized_sentence) <= max_len:
                # Sentence fits within max_len, add as is
                final_chunks.append({"text": sanitized_sentence, "is_para_end": is_paragraph_end})
            else:
                # Sentence is too long, split by words
                words = sanitized_sentence.split()
                current_sub_chunk_words = []
                for word_index, word in enumerate(words):
                    is_last_word_of_sentence = (word_index == len(words) - 1)
                    next_len = len(" ".join(current_sub_chunk_words + [word]))
                    if next_len > max_len and current_sub_chunk_words:
                        # Current chunk is full, add it
                        final_chunks.append({"text": " ".join(current_sub_chunk_words), "is_para_end": False})
                        current_sub_chunk_words = [word]
                    else:
                        current_sub_chunk_words.append(word)
                
                if current_sub_chunk_words:
                    # Add the last chunk, mark paragraph end if appropriate
                    final_chunks.append({"text": " ".join(current_sub_chunk_words), "is_para_end": is_paragraph_end})
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

            # Add silence (pause) after the sentence
            silence = np.zeros(int(pause_duration * sample_rate), dtype=np.int16)
            final_wav_data = np.concatenate([int16_wav_data, silence])
            
            # Write the audio chunk to a WAV file
            write_wav(output_wav_path, sample_rate, final_wav_data)
            
            # Calculate the duration in seconds (for LRC timing)
            duration_sec = len(final_wav_data) / sample_rate
            # Send result back to main process
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
    - Sets up output folders
    - Starts the TTS worker process
    - Extracts chapters from the EPUB
    - Splits each chapter into chunks
    - Sends jobs to the worker and collects results
    - Assembles the final MP3 and LRC files for each chapter
    - Embeds synchronized lyrics into the MP3 metadata
    """
    start_time = time.time()
    
    # Prepare output directories for the book and audio chapters
    book_output_dir = os.path.join(os.getcwd(), args.book_title)
    audio_chapters_dir = os.path.join(book_output_dir, "audio_chapters")
    os.makedirs(audio_chapters_dir, exist_ok=True)

    # --- Extract chapters from the EPUB file FIRST ---
    chapters = extract_chapters_from_epub(args.epub_file, args.skip_start, args.skip_end)
    total_chapters = len(chapters)

    # --- Now load or create the progress file ---
    progress_path = os.path.join(book_output_dir, "progress.json")
    progress_data = load_or_create_progress_file(progress_path, chapters)

    print("Initializing lightweight text splitter...")
    splitter_tts = TTS(args.tts_model_name)  # Used only for sentence splitting, not for audio
    print("Splitter ready.")
    
    print("\n--- Starting Persistent Worker Process ---")
    job_queue = Queue(maxsize=64)      # Queue for sending jobs to the worker
    results_queue = Queue(maxsize=64)  # Queue for receiving results from the worker
    consumer_process = Process(target=tts_consumer, args=(job_queue, results_queue, args))
    consumer_process.start()

    # Extract chapters from the EPUB file
    chapters = extract_chapters_from_epub(args.epub_file, args.skip_start, args.skip_end)
    total_chapters = len(chapters)
    
    for chapter_info in chapters:
        chapter_num = chapter_info["num"]
        chapter_title = chapter_info["title"]
        chapter_text = chapter_info["text"]

        # Check progress file for this chapter's status
        chapter_progress = next((c for c in progress_data["chapters"] if c["num"] == chapter_num), None)
        if chapter_progress and chapter_progress.get("status") == "done":
            print(f"Chapter {chapter_num} already marked as done in progress file. Skipping.")
            continue

        # Mark as in_progress before starting
        update_progress_file(progress_path, chapter_num, "in_progress")

        chapter_audio_path = os.path.join(audio_chapters_dir, f"chapter_{chapter_num:04d}.mp3")
        
        print(f"\n>>> Processing Chapter {chapter_num}/{total_chapters}: {chapter_title}")

        # Skip chapter if already processed and not forcing reprocess
        if not args.force_reprocess and os.path.exists(chapter_audio_path):
            print("Output file already exists. Skipping.")
            continue

        # Create a temporary folder for storing audio chunks for this chapter
        temp_chunk_folder = os.path.join(book_output_dir, "temp_audio_chunks")
        if os.path.exists(temp_chunk_folder):
            shutil.rmtree(temp_chunk_folder)  # Remove old temp folder if it exists
        os.makedirs(temp_chunk_folder)
        
        # Combine chapter title and text for TTS (so title is read aloud)
        full_text_with_title = f"{chapter_title}\n\n{chapter_text}"
        # Split the chapter into manageable chunks for TTS
        sentence_chunks = robust_sentence_splitter(full_text_with_title, splitter_tts.synthesizer, args.max_len)
        total_chunks = len(sentence_chunks)

        # Prepare lists to store timestamps and file paths for each chunk
        chapter_timestamps = [None] * total_chunks
        sentence_files_ordered = [None] * total_chunks

        # --- Collector thread: collects results from worker process ---
        def collect_results():
            """
            Runs in a separate thread.
            Collects results from the worker process and stores them in order.
            """
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

        # --- Producer: sends jobs to worker process ---
        for i, chunk_info in enumerate(sentence_chunks):
            sentence_text = chunk_info["text"]
            is_para_end = chunk_info["is_para_end"]
            # Use a longer pause at paragraph ends
            pause_duration = args.para_pause if is_para_end else args.pause
            output_wav_path = os.path.join(temp_chunk_folder, f"s_{i:04d}.wav")
            # Send job to the worker process
            job_queue.put((i, sentence_text, output_wav_path, pause_duration))
            print(f"\r  > [Producer] Sent job {i+1}/{total_chunks} to queue.", end="")

        print("\n  > [Producer] All jobs sent. Waiting for results...")
        collector_thread.join()  # Wait for all results to be collected
        
        print("\n  > Assembling and finalizing chapter MP3...")
        
        # --- Add extra silence between chunks for robustness ---
        sample_rate = splitter_tts.synthesizer.output_sample_rate
        silence_duration_samples = int(args.pause * sample_rate)
        silence_wav = np.zeros(silence_duration_samples, dtype=np.int16)
        silence_file_path = os.path.join(temp_chunk_folder, "silence.wav")
        write_wav(silence_file_path, sample_rate, silence_wav)

        # --- Create filelist for ffmpeg concat ---
        filelist_path = os.path.join(temp_chunk_folder, "filelist.txt")
        with open(filelist_path, 'w', encoding='utf-8') as f:
            for i, wav_file in enumerate(sentence_files_ordered):
                if wav_file and os.path.exists(wav_file):
                    f.write(f"file '{os.path.basename(wav_file)}'\n")
                    # Add a silence file after every chunk except the last one
                    if i < len(sentence_files_ordered) - 1:
                        f.write(f"file '{os.path.basename(silence_file_path)}'\n")
        
        # --- Use ffmpeg to concatenate all wavs and convert to mp3 ---
        ffmpeg_command = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', os.path.basename(filelist_path),
            '-c', 'libmp3lame', '-b:a', '192k', os.path.abspath(chapter_audio_path)
        ]
        try:
            subprocess.run(ffmpeg_command, check=True, capture_output=True, cwd=temp_chunk_folder)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] ffmpeg failed while assembling chapter. See logs above.")
            continue

        # --- Generate LRC (lyrics) file for chapter ---
        chapter_lrc_path = chapter_audio_path.replace('.mp3', '.lrc')
        lrc_lines, current_time = [], 0.0
        for entry in chapter_timestamps:
            if entry and entry['text'] and entry['duration'] > 0:
                lrc_lines.append(f"{format_lrc_timestamp(current_time)}{entry['text']}")
                current_time += entry['duration']
        with open(chapter_lrc_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lrc_lines))
        print(f"  > Saved LRC lyrics to: {os.path.basename(chapter_lrc_path)}")

        # --- Embed LRC as SYLT (synchronized lyrics) in MP3 metadata ---
        try:
            audio = ID3(chapter_audio_path)
        except Exception:
            audio = ID3()
            
        lrc_text_for_embedding = "\n".join(lrc_lines)
        lrc_entries, lrc_pattern = [], re.compile(r"\[(\d+):(\d+)\.(\d+)\](.*)")
        for line in lrc_text_for_embedding.splitlines():
            match = lrc_pattern.match(line)
            if match:
                minutes, seconds, hundredths = int(match.group(1)), int(match.group(2)), int(match.group(3))
                timestamp_ms = (minutes * 60 + seconds) * 1000 + hundredths * 10
                lyric = match.group(4).strip()
                lrc_entries.append((lyric, timestamp_ms))
        
        sylt = SYLT(encoding=Encoding.UTF8, lang='eng', format=2, type=1, desc='', text=lrc_entries)
        audio.setall('SYLT', [sylt])
        audio.save(v2_version=3)  # Save with v2.3 for max compatibility
        print(f"  > Embedded lyrics directly into {os.path.basename(chapter_audio_path)}.")

        # --- Clean up temp files ---
        shutil.rmtree(temp_chunk_folder)
        print(f"--- Successfully completed and finalized Chapter {chapter_num} ---")

        # Mark as done after finishing
        update_progress_file(progress_path, chapter_num, "done")

    # --- All chapters done ---
    print("\n--- All chapters processed! Shutting down worker process... ---")
    job_queue.put("STOP")  # Tell the worker process to exit
    consumer_process.join()  # Wait for worker to finish
    
    print("\n--- Project Complete! ---")
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time/3600:.2f} hours.")
    print(f"Your chapterized audiobook is ready in: '{audio_chapters_dir}'")

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
    
    parser.add_argument("--skip_start", type=int, default=6, help="Number of 'chapters' to skip at the beginning of the EPUB.") # Change this to any number of chapter according to your book to skip the preface and introduction
    parser.add_argument("--skip_end", type=int, default=1, help="Number of 'chapters' to skip at the end of the EPUB.") # Change this to any number of chapter according to your book to skip the appendix or afterword
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds of silence to add between sentences.")
    parser.add_argument("--para_pause", type=float, default=1.2, help="A longer pause in seconds for paragraph breaks.")
    parser.add_argument("--max_len", type=int, default=240, help="Maximum character length for a single text chunk.") # Do not change this value this is the maximum length of a single text chunk that the TTS model can handle, if you change this value it will break the code, you decrease this value if you want to split the text into smaller chunks, but it is not recommended to increase this value.
    parser.add_argument("--temperature", type=float, default=0.8, help="TTS generation temperature.") # Change this value to control the randomness of the TTS generation, lower values make it more deterministic, higher values make it more creative.
    parser.add_argument("--top_p", type=float, default=0.8, help="TTS generation top_p.") # Change this value to control the diversity of the TTS generation, lower values make it more deterministic, higher values make it more creative.
    
    # Do not change these values unless you know what you are doing 
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

