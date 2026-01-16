"""
Unit tests for the Broadcast Audio Prep Tool.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import prep_audio


class TestVersion:
    """Tests for version information."""

    def test_version_exists(self):
        """Verify __version__ is defined."""
        assert hasattr(prep_audio, "__version__")
        assert isinstance(prep_audio.__version__, str)

    def test_version_format(self):
        """Verify version follows semver format."""
        parts = prep_audio.__version__.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts[:2])


class TestModes:
    """Tests for processing mode configurations."""

    def test_modes_exist(self):
        """Verify both modes are defined."""
        assert "BG_PREP" in prep_audio.MODES
        assert "QC_MASTER" in prep_audio.MODES

    def test_bg_prep_config(self):
        """Verify BG_PREP mode configuration."""
        mode = prep_audio.MODES["BG_PREP"]
        assert mode["TARGET_LKFS"] == -27.0
        assert mode["TP_LIMIT"] == -6.0
        assert "DESC" in mode

    def test_qc_master_config(self):
        """Verify QC_MASTER mode configuration."""
        mode = prep_audio.MODES["QC_MASTER"]
        assert mode["TARGET_LKFS"] == -24.0
        assert mode["TP_LIMIT"] == -2.0
        assert "DESC" in mode


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_true_peak_db(self):
        """Test true-peak calculation."""
        # Create a simple sine wave at known amplitude
        sample_rate = 48000
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
        audio = audio.reshape(-1, 1)

        peak_db = prep_audio.get_true_peak_db(audio, sample_rate)

        # 0.5 amplitude = -6.02 dB, but with 4x oversampling peaks may vary slightly
        assert -8.0 < peak_db < -4.0

    def test_apply_uniform_gain(self):
        """Test gain application."""
        audio = np.array([[0.5], [0.25], [-0.5]])

        # Apply +6 dB gain (approximately 2x)
        gained = prep_audio.apply_uniform_gain(audio, 6.0)

        assert gained[0, 0] > 0.9  # Should be close to 1.0
        assert gained[1, 0] > 0.45  # Should be close to 0.5
        assert gained[2, 0] < -0.9  # Should be close to -1.0

    def test_limit_ceiling(self):
        """Test ceiling limiter."""
        # Audio with peaks at 0.9
        audio = np.array([[0.9], [-0.9], [0.5]])

        # Apply -6 dB ceiling (0.5)
        limited, reduction = prep_audio.limit_ceiling(audio, -6.0)

        assert np.max(np.abs(limited)) <= 0.52  # Should be at or below ceiling
        assert reduction < 0  # Should have applied reduction

    def test_limit_ceiling_no_reduction_needed(self):
        """Test ceiling limiter when audio is already below ceiling."""
        audio = np.array([[0.1], [-0.1], [0.05]])

        limited, reduction = prep_audio.limit_ceiling(audio, -6.0)

        np.testing.assert_array_equal(limited, audio)
        assert reduction == 0.0


class TestFileIterator:
    """Tests for file iteration."""

    def test_iter_audio_files_wav(self, input_dir_with_files):
        """Test finding WAV files."""
        files = list(prep_audio.iter_audio_files(str(input_dir_with_files)))
        assert len(files) == 2
        assert all(f.endswith(".wav") for f in files)

    def test_iter_audio_files_empty_dir(self, empty_dir):
        """Test with empty directory."""
        files = list(prep_audio.iter_audio_files(str(empty_dir)))
        assert len(files) == 0

    def test_iter_audio_files_recursive(self, temp_dir, sample_wav):
        """Test recursive file iteration."""
        # Create nested structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        import shutil
        shutil.copy(sample_wav, temp_dir / "root.wav")
        shutil.copy(sample_wav, subdir / "nested.wav")

        # Non-recursive should find 1
        files_flat = list(prep_audio.iter_audio_files(str(temp_dir), recursive=False))
        assert len(files_flat) == 1

        # Recursive should find 2
        files_recursive = list(prep_audio.iter_audio_files(str(temp_dir), recursive=True))
        assert len(files_recursive) == 2


class TestProcessOneFile:
    """Tests for single file processing."""

    def test_process_file_bg_prep(self, sample_wav):
        """Test processing a file in BG_PREP mode."""
        mode_config = prep_audio.MODES["BG_PREP"]
        processed, sr, report = prep_audio.process_one_file(str(sample_wav), mode_config)

        assert sr == 48000  # Should be resampled to target
        assert processed.shape[1] == 2  # Stereo
        assert "filename" in report
        assert "status" in report
        assert "post_lufs" in report

    def test_process_file_qc_master(self, sample_wav):
        """Test processing a file in QC_MASTER mode."""
        mode_config = prep_audio.MODES["QC_MASTER"]
        processed, sr, report = prep_audio.process_one_file(str(sample_wav), mode_config)

        assert sr == 48000
        assert "post_lufs" in report
        # Post LUFS should be close to target
        assert -25.0 < report["post_lufs"] < -23.0

    def test_process_file_resampling(self, sample_wav):
        """Test that files are resampled to 48kHz."""
        # sample_wav is 44.1kHz
        mode_config = prep_audio.MODES["BG_PREP"]
        processed, sr, report = prep_audio.process_one_file(str(sample_wav), mode_config)

        assert sr == 48000
        assert report["resampled"] == "Yes"


class TestDeterministicDithering:
    """Tests for deterministic dithering."""

    def test_dithering_reproducibility(self, temp_dir):
        """Test that same file produces identical output with same seed."""
        filepath1 = temp_dir / "output1.wav"
        filepath2 = temp_dir / "output2.wav"

        audio = np.random.rand(1000, 2).astype(np.float32) * 0.5
        seed = 12345

        prep_audio.write_wav_24bit_tpdf(str(filepath1), audio, 48000, seed)
        prep_audio.write_wav_24bit_tpdf(str(filepath2), audio, 48000, seed)

        data1, _ = sf.read(str(filepath1))
        data2, _ = sf.read(str(filepath2))

        np.testing.assert_array_equal(data1, data2)


class TestHTMLReport:
    """Tests for HTML report generation."""

    def test_generate_html_report(self, temp_dir):
        """Test HTML report generation."""
        import pandas as pd

        data = [
            {"filename": "test1.wav", "status": "PASS", "post_lufs": -27.0},
            {"filename": "test2.wav", "status": "FAIL (LUFS)", "post_lufs": -30.0},
        ]
        df = pd.DataFrame(data)
        html_path = temp_dir / "report.html"

        prep_audio.generate_html_report(df, str(html_path), "Test Mode")

        assert html_path.exists()
        content = html_path.read_text()
        assert "test1.wav" in content
        assert "PASS" in content
        assert "Test Mode" in content
