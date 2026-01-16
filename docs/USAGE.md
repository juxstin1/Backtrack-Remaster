# Usage Guide

This guide provides detailed examples and workflows for using the Broadcast Audio Prep Tool.

## Quick Start

```bash
# Basic usage - process files in ./input, output to ./output
python prep_audio.py

# Specify input and output directories
python prep_audio.py --input /path/to/audio --output /path/to/processed

# Use QC_MASTER mode for final delivery
python prep_audio.py --mode QC_MASTER
```

## Command Reference

```
python prep_audio.py [OPTIONS]

Options:
  --input PATH          Input directory containing audio files
  --output PATH         Output directory for processed files
  --mode {BG_PREP,QC_MASTER}  Processing mode (default: BG_PREP)
  --suffix              Add descriptive suffix to output filenames
  --dry-run             Preview without writing files
  -v, --verbose         Enable debug output
  -q, --quiet           Suppress non-essential output
  -r, --recursive       Scan subdirectories for audio files
  -j, --jobs N          Number of parallel processing workers
  --version             Show version and exit
  -h, --help            Show help message
```

## Workflow Examples

### 1. Background Music Preparation

For music beds that will sit under dialogue:

```bash
python prep_audio.py \
    --input ./music_beds \
    --output ./prepped_music \
    --mode BG_PREP \
    --suffix
```

Output files will have names like:
```
track_01__BG_PREP__-27LUFS_-6dBTP.wav
```

### 2. Final Master QC

For delivery-ready masters:

```bash
python prep_audio.py \
    --input ./masters \
    --output ./delivery \
    --mode QC_MASTER
```

### 3. Preview Before Processing

Use dry-run to see what would happen without writing files:

```bash
python prep_audio.py --input ./audio --dry-run

# Output:
# [DRY RUN] Preview mode - no files will be written.
# Processing: 100%|████████| 10/10 [00:05<00:00]
#
# [DRY RUN] Summary:
#   Files to process: 10
#   Mode: Background Music Prep (-27 LUFS / -6 dBTP)
#   Expected pass: 10/10
```

### 4. Process Large Batches in Parallel

For large batches, use multiple workers:

```bash
python prep_audio.py \
    --input ./large_batch \
    --output ./processed \
    --jobs 4
```

### 5. Process Nested Folder Structure

To process files in subdirectories:

```bash
python prep_audio.py \
    --input ./project \
    --output ./processed \
    --recursive
```

### 6. Quiet Mode for Scripts

When running in automated pipelines:

```bash
python prep_audio.py --input ./audio --quiet
```

### 7. Verbose Mode for Debugging

To see detailed processing information:

```bash
python prep_audio.py --input ./audio --verbose
```

## Understanding Reports

After processing, three report files are generated in the `Reports` subdirectory:

### qc_report.csv

Spreadsheet-compatible format with columns:
- `filename` - Original filename
- `pre_lufs` - Loudness before processing
- `post_lufs` - Loudness after processing
- `pre_tp_db` - True peak before processing
- `post_tp_db` - True peak after processing
- `gain_db` - Applied gain
- `limiter_gr_db` - Limiter gain reduction
- `resampled` - Whether resampling was needed
- `status` - PASS, WARN, or FAIL
- `duration_s` - File duration
- `channels` - Channel count

### summary_report.html

Color-coded HTML report for easy review:
- 🟢 Green rows: PASS
- 🟡 Yellow rows: WARN (Peak Corrected)
- 🔴 Red rows: FAIL

### qc_report.json

Machine-readable JSON format including:
- Version information
- Processing parameters
- Summary statistics
- Per-file details

## Status Meanings

| Status | Meaning |
|--------|---------|
| PASS | File meets all specifications |
| WARN (Peak Corrected) | Post-write peak exceeded limit, automatically corrected |
| FAIL (LUFS) | Loudness outside ±0.5 LU of target |
| FAIL (Peak) | True peak exceeds limit |

## Best Practices

### Before Processing

1. **Organize source files**: Group files by intended use (background music, masters, etc.)
2. **Check source quality**: Ensure files aren't already heavily limited or distorted
3. **Preview first**: Use `--dry-run` to check expected results

### During Processing

1. **Use appropriate mode**: BG_PREP for music beds, QC_MASTER for finals
2. **Monitor progress**: Watch for WARN/FAIL status in progress bar
3. **Check available disk space**: Output files may be larger than inputs

### After Processing

1. **Review reports**: Check the HTML summary for any warnings or failures
2. **Spot-check audio**: Listen to a few processed files for quality
3. **Archive reports**: Keep the JSON report for project documentation

## Troubleshooting

### "No WAV or AIFF files found"

- Check the input directory path
- Ensure files have correct extensions (.wav, .aif, .aiff)
- Use `--recursive` if files are in subdirectories

### "Source file is silent"

- The file contains no audio data or is essentially silent
- Check the source file in an audio editor

### Memory errors on large files

- Process files individually
- Use a system with more RAM
- Consider splitting very long files

### Slow processing

- Use `--jobs 4` for parallel processing
- Ensure source files aren't on a network drive
- Check available CPU resources

## Integration Examples

### Shell Script Batch Processing

```bash
#!/bin/bash
for project in ./projects/*/; do
    python prep_audio.py \
        --input "$project/audio" \
        --output "$project/processed" \
        --mode QC_MASTER \
        --quiet
done
```

### Python Integration

```python
import subprocess
import json

result = subprocess.run([
    "python", "prep_audio.py",
    "--input", "./audio",
    "--output", "./processed",
    "--quiet"
], capture_output=True)

# Read the JSON report
with open("./processed/Reports/qc_report.json") as f:
    report = json.load(f)
    print(f"Processed {report['summary']['total']} files")
    print(f"Passed: {report['summary']['passed']}")
```
