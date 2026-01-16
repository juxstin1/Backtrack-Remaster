---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Run command `python prep_audio.py ...`
2. With input file type '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Error Output
```
Paste any error messages or traceback here
```

## Environment
- **OS**: [e.g., Windows 11, macOS 14, Ubuntu 22.04]
- **Python version**: [e.g., 3.11.0]
- **Package versions**:
  ```
  pip freeze | grep -E "(soundfile|pyloudnorm|scipy|pandas|numpy)"
  ```

## Audio File Details
- **Format**: [e.g., WAV, AIFF]
- **Sample rate**: [e.g., 44100, 48000]
- **Bit depth**: [e.g., 16-bit, 24-bit, 32-bit float]
- **Channels**: [e.g., mono, stereo]
- **Duration**: [e.g., 30 seconds]

## Command Used
```bash
python prep_audio.py --mode BG_PREP --input ./my_folder
```

## Additional Context
Add any other context about the problem here.

## Checklist
- [ ] I have checked that this issue hasn't already been reported
- [ ] I have included the full error traceback
- [ ] I have tested with a different audio file to confirm it's not file-specific
