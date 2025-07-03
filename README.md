# The AI Audiobook Factory  Audiobook

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

This project is a complete, end-to-end Python-based pipeline for converting e-books (in EPUB format) into high-quality, chapterized audiobooks with synchronised, sentence-level lyrics.

It uses the powerful open-source [Coqui XTTS v2](https://github.com/coqui-ai/TTS) model for expressive, natural-sounding narration through voice cloning. The entire process is managed by a robust, parallel-processing script that can handle massive books and is fully resumable, meaning you can stop and restart long generation jobs without losing progress.

## Key Features

- **High-Quality TTS:** Leverages the Coqui XTTS v2 model for expressive, cloned-voice narration based on a short audio sample.
- **EPUB to Audiobook:** Directly processes `.epub` files, automatically parsing chapters, titles, and cleaning text content.
- **Persistent Worker Architecture:** Initialises the large AI model only once, making chapter-to-chapter processing significantly faster and avoiding redundant loading.
- **Parallel Processing:** Uses a producer-consumer pipeline to keep the GPU fully utilised, maximising generation speed for each chapter.
- **Fully Resumable:** If the process is interrupted, it automatically detects and skips already completed chapters on the next run, saving hours of work.
- **Chapterized Output:** Creates a separate, neatly named MP3 file for each chapter, perfect for use in modern audiobook players.
- **Synchronised Lyrics:** Automatically generates a timed `.lrc` file for each chapter and embeds the data directly into the MP3's ID3 tags for compatible players.
- **Fully Configurable:** All major settings (file paths, TTS parameters, chapter skipping) are handled via command-line arguments for maximum flexibility.
- **Utility Toolkit:** Includes scripts for:
    - **`test_bench.py`:** Quickly test narrator voices and TTS parameters on a small text sample.
    - **`lrc_to_srt_converter.py`:** Convert the generated LRC lyric files to the SRT subtitle format for use in video players.
    - **`finalizer.py`:** A legacy tool to upgrade audiobooks created with older versions of this script.

## Setup

### 1. Prerequisites
- Python 3.10 or higher.
- An **NVIDIA GPU with at least 8GB of VRAM** is **highly recommended** for acceptable performance.
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (version 11.8 or higher is recommended for library compatibility).
- **FFmpeg:** This is essential for combining audio chunks and creating the final MP3 files. Please take a look at the installation instructions below.

---
### **FFmpeg Installation**

The script needs `ffmpeg` to be accessible from the command line. You have two options:

#### **Option 1: System-wide Installation (Recommended)**
This makes `ffmpeg` available to any program on your computer.

*   **On Windows (using a package manager):**
    *   Open PowerShell and use either Winget (built-in) or Chocolatey (if installed).
    *   **Winget:** `winget install -e --id Gyan.FFmpeg`
    *   **Chocolatey:** `choco install ffmpeg`

*   **On Windows (Manual):**
    1.  Download a build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or the [official site](https://ffmpeg.org/download.html).
    2.  Unzip the downloaded file to a permanent location, for example `C:\ffmpeg`.
    3.  Search for "Environment Variables" in the Start Menu and select "Edit the system environment variables".
    4.  Click the "Environment Variables..." button.
    5.  Under "System variables", find the `Path` variable, select it, and click "Edit...".
    6.  Click "New" and add the full path to the `bin` folder inside your ffmpeg directory (e.g., `C:\ffmpeg\bin`).
    7.  Click OK on all windows to close them. **You must close and re-open your terminal/PowerShell for this change to take effect.**

*   **On macOS (using Homebrew):**
    ```bash
    brew install ffmpeg
    ```

*   **On Linux (Debian/Ubuntu):**
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```

*   **On Linux (Fedora 22+):**
    ```bash
    sudo dnf update && sudo dnf install ffmpeg
    ```    

**To verify your installation,** open a **new** terminal and type `ffmpeg -version`. If it prints version information, you are ready.

#### **Option 2: Project-specific (Portable)**
If you do not want to install FFmpeg system-wide, you can place it directly in this project folder.
1.  Download the FFmpeg build as described in the manual Windows instructions.
2.  Unzip the file.
3.  Go into the `bin` folder.
4.  Copy **all** the files from inside `bin` (`ffmpeg.exe`, `ffprobe.exe`, `ffplay.exe`, and all the `.dll` files) and paste them directly into the root of your `Audio_Book_Maker` folder.

---
### 2. Installation
1.  Clone this repository or download the source code into your `Audio_Book_Maker` folder.
2.  It is **strongly recommended** to use a Python virtual environment. Open a terminal in the project folder and run:
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate the environment
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
3.  Install the required Python libraries using the provided `requirements.txt` file. This includes a specific, GPU-enabled version of PyTorch.
    ```bash
    pip install -r requirements.txt
    ```

## How to Use Your Audiobook Factory

### Step 1: Prepare Your Assets
1.  Place your e-book file (e.g., `MyBook.epub`) into the Your_Novel folder.
2.  Find a high-quality, clean (no music/noise), 15-30 second audio clip of the narrator voice you want to use. Save it as a `.wav` file (e.g., `MyNarrator.wav`) inside a `narrator_voice` folder.

### Step 2 (Recommended): The Test Bench
Before starting a multi-hour generation, find the perfect voice style.
1.  Create a small `test.txt` file with a few paragraphs from your book.
2.  Open `test_bench.py` and update the `INPUT_TEXT_FILE` and `NARRATOR_VOICE_SAMPLE` paths at the top.
3.  Adjust the `TEMPERATURE` and `TOP_P` values to experiment with expressiveness vs. stability.
4.  Run the script: `python test_bench.py`.
5.  Listen to the `test_output.wav` file. Repeat until you are happy with the style and note the best parameter values.

### Step 3: Run the Main Production
This is the main command you will run. It will create a new project folder for your audiobook and begin processing chapter by chapter.

Open your terminal, ensure your virtual environment is activated, and run the `audiobook_factory.py` script with your desired settings.

**Example Command (for Windows):**
```bash
python audiobook_factory.py ^
  --epub_file "./Your_Novel/MyBook.epub" ^
  --voice_file "./narrator_voice/MyNarrator.wav" ^
  --book_title "My Awesome Audiobook" ^
  --skip_start 6 ^
  --skip_end 2 ^
  --temperature 0.8 ^
  --top_p 0.8
```

The script will start, initialise the worker process (this may take a few minutes), and then begin processing. You can stop this process at any time and re-run the exact same command to resume where you left off.

### Step 4: Enjoy!
Once the script is finished, you will find a new folder named `My Awesome Audiobook/`. Inside, the `audio_chapters/` subfolder will contain all of your final, per-chapter MP3 files with embedded synchronised lyrics, ready to be enjoyed in your favorite audiobook player.

## Project File Overview
-   `audiobook_factory.py`: The main, unified script. This is the one you run to generate a full audiobook from an EPUB.
-   `test_bench.py`: A utility for quickly testing narrator voices and TTS tuning parameters. Run this before the main script for testing.
-   `lrc_to_srt_converter.py`: A utility to convert the generated `.lrc` lyric files into `.srt` subtitle files.
-   `requirements.txt`: A list of all the Python libraries required to run this project.
