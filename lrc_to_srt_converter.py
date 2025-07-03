import os
import glob
import re

# --- CONFIGURATION ---

LRC_DIRECTORY = "./<Your_Book>/audio_chapters" # In this the <Your_Book> folder will be created by the audiobook_factory.py script.

# 2. How long should the very last subtitle in each file stay on screen? (in seconds)
DEFAULT_LAST_LINE_DURATION = 5.0

# -----------------------------------------------------------------------------

def ms_to_srt_time(total_milliseconds):
    """Converts total milliseconds into the SRT time format HH:MM:SS,ms"""
    if total_milliseconds < 0:
        total_milliseconds = 0
    hours = int(total_milliseconds // 3600000)
    minutes = int((total_milliseconds % 3600000) // 60000)
    seconds = int((total_milliseconds % 60000) // 1000)
    milliseconds = int(total_milliseconds % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def convert_lrc_to_srt(lrc_path):
    """
    Reads an LRC file, parses its content, and writes a corresponding SRT file.
    """
    print(f"  > Converting '{os.path.basename(lrc_path)}'...")
    
    try:
        with open(lrc_path, 'r', encoding='utf-8') as f:
            lrc_lines = f.readlines()

        # Regex to capture [mm:ss.xx] timestamps and the text
        lrc_pattern = re.compile(r"\[(\d+):(\d+)\.(\d+)\](.*)")
        parsed_lines = []

        for line in lrc_lines:
            match = lrc_pattern.match(line)
            if match:
                minutes, seconds, hundredths, text = match.groups()
                start_ms = (int(minutes) * 60 + int(seconds)) * 1000 + int(hundredths) * 10
                # Clean up the text, removing extra whitespace
                text = text.strip()
                if text: # Only add lines that have actual text
                    parsed_lines.append({'start_ms': start_ms, 'text': text})

        if not parsed_lines:
            print("    - No valid LRC lines found. Skipping.")
            return

        # Build the SRT blocks
        srt_blocks = []
        for i, current_line in enumerate(parsed_lines):
            start_time_str = ms_to_srt_time(current_line['start_ms'])
            
            # Determine the end time
            if i + 1 < len(parsed_lines):
                # End time is the start time of the next line
                end_time_ms = parsed_lines[i + 1]['start_ms']
            else:
                # For the last line, add the default duration
                end_time_ms = current_line['start_ms'] + int(DEFAULT_LAST_LINE_DURATION * 1000)

            end_time_str = ms_to_srt_time(end_time_ms)
            
            # Assemble the SRT block
            block_index = i + 1
            text = current_line['text']
            srt_block = f"{block_index}\n{start_time_str} --> {end_time_str}\n{text}\n"
            srt_blocks.append(srt_block)
        
        # Write the SRT file
        srt_path = lrc_path.replace('.lrc', '.srt')
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(srt_blocks))
            
        print(f"    - Success! Saved to '{os.path.basename(srt_path)}'")

    except Exception as e:
        print(f"    - ERROR converting file: {e}")


def main():
    """Main function to find and convert all LRC files in the target directory."""
    print("--- LRC to SRT Converter ---")
    
    # Create the full path to search for .lrc files
    search_path = os.path.join(LRC_DIRECTORY, "*.lrc")
    lrc_files = sorted(glob.glob(search_path))

    if not lrc_files:
        print(f"No .lrc files found in the specified directory: '{LRC_DIRECTORY}'")
        return

    print(f"Found {len(lrc_files)} LRC files to convert.\n")

    for lrc_path in lrc_files:
        convert_lrc_to_srt(lrc_path)

    print("\n--- Conversion Complete! ---")


if __name__ == "__main__":
    if not os.path.exists(LRC_DIRECTORY):
        print(f"ERROR: The specified directory does not exist: '{LRC_DIRECTORY}'")
        exit(1)
    main()