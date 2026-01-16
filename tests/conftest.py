"""
Pytest configuration and fixtures for Backtrack-Remaster tests.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_wav(temp_dir):
    """Create a sample WAV file for testing."""
    filepath = temp_dir / "test_audio.wav"

    # Generate 2 seconds of stereo audio at 44.1kHz
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create a simple sine wave (440 Hz) with some amplitude variation
    left_channel = 0.5 * np.sin(2 * np.pi * 440 * t)
    right_channel = 0.5 * np.sin(2 * np.pi * 880 * t)
    audio = np.column_stack([left_channel, right_channel]).astype(np.float32)

    sf.write(str(filepath), audio, sample_rate, subtype="PCM_16")
    return filepath


@pytest.fixture
def sample_wav_loud(temp_dir):
    """Create a loud WAV file that needs limiting."""
    filepath = temp_dir / "loud_audio.wav"

    sample_rate = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create loud audio (near 0 dBFS)
    audio = 0.95 * np.sin(2 * np.pi * 1000 * t)
    audio = np.column_stack([audio, audio]).astype(np.float32)

    sf.write(str(filepath), audio, sample_rate, subtype="PCM_16")
    return filepath


@pytest.fixture
def sample_wav_quiet(temp_dir):
    """Create a quiet WAV file that needs normalization."""
    filepath = temp_dir / "quiet_audio.wav"

    sample_rate = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create quiet audio (around -40 dBFS)
    audio = 0.01 * np.sin(2 * np.pi * 440 * t)
    audio = np.column_stack([audio, audio]).astype(np.float32)

    sf.write(str(filepath), audio, sample_rate, subtype="PCM_16")
    return filepath


@pytest.fixture
def sample_aiff(temp_dir):
    """Create a sample AIFF file for testing."""
    filepath = temp_dir / "test_audio.aiff"

    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio = np.column_stack([audio, audio]).astype(np.float32)

    sf.write(str(filepath), audio, sample_rate, format="AIFF")
    return filepath


@pytest.fixture
def empty_dir(temp_dir):
    """Create an empty directory."""
    empty = temp_dir / "empty"
    empty.mkdir()
    return empty


@pytest.fixture
def input_dir_with_files(temp_dir, sample_wav):
    """Create an input directory with audio files."""
    input_dir = temp_dir / "input"
    input_dir.mkdir()

    # Copy sample wav to input dir
    import shutil
    shutil.copy(sample_wav, input_dir / "test1.wav")

    # Create another file
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 660 * t)
    audio = np.column_stack([audio, audio]).astype(np.float32)
    sf.write(str(input_dir / "test2.wav"), audio, sample_rate, subtype="PCM_16")

    return input_dir


@pytest.fixture
def output_dir(temp_dir):
    """Create an output directory."""
    output = temp_dir / "output"
    output.mkdir()
    return output
