
# Broadcast Audio Prep Tool

This project provides a command-line tool for preparing WAV or AIFF audio files for broadcast use. It supports two standard delivery modes:

| Mode        | Target Loudness | True-Peak Limit | Typical use case                         |
|-------------|-----------------|-----------------|------------------------------------------|
| **BG_PREP** | -27 LUFS        | -6 dBTP         | Music beds that will sit under dialogue  |
| **QC_MASTER** | -24 LUFS     | -2 dBTP         | Final masters for delivery               |

The script applies ITU-R BS.1770 loudness normalisation, resamples to 48 kHz using a high quality polyphase filter, applies a true-peak limiter, and exports 24-bit PCM with deterministic triangular dither. It also generates CSV and HTML reports summarising the processing results.

## Installation

1. **Install Python.** Ensure Python 3.8 or later is available:

   ```bash
   python --version
   ```

2. **Install dependencies.** From a terminal:

   ```bash
   pip install soundfile pyloudnorm scipy pandas numpy
   ```

   On some platforms you may also need to install `libsndfile` via your package manager (for example `apt`, `brew`, or `choco`).

## Usage

1. **Drop your source files** into the project's `input\` folder. Only `.wav`, `.aif`, or `.aiff` files in that folder are processed.

2. **Run the script** with Python:

   ```bash
   python prep_audio.py
   ```
   With no extra flags the script reads from `input\` and writes masters to `output\`. A `Reports` subdirectory is created in `output\` containing `qc_report.csv`, `summary_report.html`, and `processing_log.txt`.

3. **Optional flags:**

   - `--mode {BG_PREP,QC_MASTER}` – choose the processing mode (defaults to `BG_PREP`).
   - `--suffix` – append a descriptive suffix (for example `__BG_PREP__-27LUFS_-6dBTP`) to each output filename.

### Example

```bash
python prep_audio.py \
    --input /home/user/background_cues \
    --output /home/user/prepped_audio \
    --mode BG_PREP \
    --suffix
```

## FAQ

**Why does the script read files back after writing them?**  
Quantising 32-bit floats to 24-bit PCM can occasionally increase the true-peak level by a fraction of a decibel. The script re-measures the final file and automatically scales it down if it exceeds the specified true-peak limit.

**What does deterministic dithering mean?**  
The dither noise generator is seeded from each filename, ensuring that running the script twice on the same file produces identical output bits—a useful property for QA and reproducibility.

**Can I use this on multichannel files?**  
Yes. The script processes files with any channel count and reports the channel count in the CSV/HTML reports. It does not downmix; each channel is left untouched aside from gain adjustments.

## Support
Refer to the inline documentation in `prep_audio.py` for more details on the processing stages and configuration options.

## Utilities
- Double-click `clear_input_folder.bat` to empty the `input\` folder between batches. The script only removes `.wav`, `.aif`, and `.aiff` files and any leftover sub-folders.

# Backtrack-Remaster
CLI tool for prepping WAV/AIFF to broadcast standards with loudness norm, true-peak limiting, dithering &amp; QC reports

