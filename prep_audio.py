"""
Broadcast Audio Prep Tool (v5.0)
================================

This script prepares WAV/AIFF audio files for broadcast use.  It supports
two modes of operation and now works directly on an input folder instead of
expecting a ZIP archive.  Processed masters are written to an output folder
alongside QC reports.
"""

import argparse
import datetime
import os
from html import escape
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly


# ---------------------------------------------------------------------------
# Configuration

MODES: Dict[str, Dict[str, float]] = {
    "BG_PREP": {
        "TARGET_LKFS": -27.0,
        "TP_LIMIT": -6.0,
        "LIM_CEIL": -7.0,
        "DESC": "Background Music Prep (-27 LUFS / -6 dBTP)",
    },
    "QC_MASTER": {
        "TARGET_LKFS": -24.0,
        "TP_LIMIT": -2.0,
        "LIM_CEIL": -3.0,
        "DESC": "QC Final Master (-24 LUFS / -2 dBTP)",
    },
}

TARGET_SR = 48_000  # Target sample rate in Hertz
PCM_SUBTYPE = "PCM_24"  # 24-bit PCM output format
VALID_EXTENSIONS = (".wav", ".aif", ".aiff")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


# ---------------------------------------------------------------------------
# Utility functions

def get_true_peak_db(samples: np.ndarray, sample_rate: int) -> float:
    """Compute the approximated true-peak level (dBTP) using 4x oversampling."""

    oversampled = resample_poly(samples, 4, 1, axis=0, window=("kaiser", 7.5))
    peak_linear = np.max(np.abs(oversampled))
    return 20.0 * np.log10(peak_linear + 1e-12)


def write_wav_24bit_tpdf(path: str, samples: np.ndarray, sample_rate: int, seed: int) -> None:
    """Write a 24-bit WAV file with deterministic TPDF dithering."""

    clipped = np.clip(samples, -1.0, 1.0)
    rng = np.random.default_rng(seed)
    dither = (
        rng.random(clipped.shape, dtype=np.float32)
        - 0.5
        + rng.random(clipped.shape, dtype=np.float32)
        - 0.5
    ) / (2**23)
    dithered = clipped + dither
    sf.write(path, dithered, sample_rate, subtype=PCM_SUBTYPE)


def enforce_tp_after_write(path: str, tp_limit_db: float) -> Tuple[bool, float, float]:
    """Verify the written file's true-peak and correct if necessary."""

    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    peak_before = get_true_peak_db(samples, sr)
    if peak_before <= tp_limit_db:
        return False, peak_before, peak_before

    margin = 10.0 ** ((tp_limit_db - peak_before) / 20.0)
    seed = abs(hash(os.path.basename(path))) % (2**32)
    write_wav_24bit_tpdf(path, samples * margin, sr, seed)
    corrected, _ = sf.read(path, dtype="float32", always_2d=True)
    peak_after = get_true_peak_db(corrected, sr)
    return True, peak_before, peak_after


def apply_uniform_gain(samples: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply a uniform gain change to the audio in dB."""

    return samples * (10.0 ** (gain_db / 20.0))


def limit_ceiling(samples: np.ndarray, ceiling_db: float) -> Tuple[np.ndarray, float]:
    """Apply a hard ceiling (brick-wall limiter) to audio data."""

    peak_linear = np.max(np.abs(samples))
    ceiling_linear = 10.0 ** (ceiling_db / 20.0)
    if peak_linear <= ceiling_linear:
        return samples, 0.0

    reduction_factor = ceiling_linear / (peak_linear + 1e-12)
    reduction_db = 20.0 * np.log10(reduction_factor + 1e-12)
    return samples * reduction_factor, reduction_db


def process_one_file(file_path: str, mode_config: Dict[str, float]) -> Tuple[np.ndarray, int, Dict[str, object]]:
    """Process a single audio file according to the given mode."""

    original_name = os.path.basename(file_path)
    data, sr = sf.read(file_path, dtype="float32", always_2d=True)

    # Remove any DC offset per channel.
    data = data - np.mean(data, axis=0, keepdims=True)

    resampled = False
    if sr != TARGET_SR:
        data = resample_poly(data, TARGET_SR, sr, axis=0, window=("kaiser", 7.5))
        sr = TARGET_SR
        resampled = True

    meter = pyln.Meter(sr)
    pre_lufs = meter.integrated_loudness(data)
    if np.isinf(pre_lufs):
        raise ValueError("Source file is silent.")

    pre_tp_db = get_true_peak_db(data, sr)
    gain_db = mode_config["TARGET_LKFS"] - pre_lufs
    processed = apply_uniform_gain(data, gain_db)
    processed, limiter_red_db = limit_ceiling(processed, mode_config["LIM_CEIL"])

    post_lufs = meter.integrated_loudness(processed)
    post_tp_db = get_true_peak_db(processed, sr)

    status = "PASS"
    if not (mode_config["TARGET_LKFS"] - 0.5 <= post_lufs <= mode_config["TARGET_LKFS"] + 0.5):
        status = "FAIL (LUFS)"
    if post_tp_db > mode_config["TP_LIMIT"]:
        status = "FAIL (Peak)"

    report = {
        "filename": original_name,
        "pre_lufs": round(pre_lufs, 2),
        "post_lufs": round(post_lufs, 2),
        "pre_tp_db": round(pre_tp_db, 2),
        "post_tp_db": round(post_tp_db, 2),
        "gain_db": round(gain_db, 2),
        "limiter_gr_db": round(limiter_red_db, 2),
        "resampled": "Yes" if resampled else "No",
        "status": status,
        "duration_s": round(data.shape[0] / sr, 2),
        "channels": data.shape[1],
    }

    return processed, sr, report


def generate_html_report(df: pd.DataFrame, path: str, mode_desc: str) -> None:
    """Write a simple, colour-coded HTML report based off the DataFrame."""

    def row_style(status: str) -> str:
        if "PASS" in status:
            return "background-color: #28a745; color: white;"
        if "WARN" in status:
            return "background-color: #ffc107; color: black;"
        return "background-color: #dc3545; color: white;"

    summary = {
        "Total": len(df),
        "Pass": len(df[df["status"] == "PASS"]),
        "Warnings": len(df[df["status"].str.contains("WARN", na=False)]),
        "Fails": len(df[df["status"].str.contains("FAIL", na=False)]),
    }

    columns = list(df.columns)
    header_cells = "".join(f"<th>{escape(str(col))}</th>" for col in columns)
    data_rows: List[str] = []
    for _, row in df.iterrows():
        style = row_style(str(row["status"]))
        cells = "".join(f"<td>{escape(str(row[col]))}</td>" for col in columns)
        data_rows.append(f'<tr style="{style}">{cells}</tr>')
    html_table = f"<table><thead><tr>{header_cells}</tr></thead><tbody>{''.join(data_rows)}</tbody></table>"

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            f"""
<html><head><title>Broadcast Audio Report</title><style>
body {{ font-family: sans-serif; margin: 2em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #f2f2f2; }}
.summary {{ margin-bottom: 2em; }}
</style></head><body>
<h1>Broadcast Audio Prep Report</h1>
<div class="summary">
  <p><b>Mode:</b> {mode_desc}</p>
  <p><b>Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  <p><b>Summary:</b> Total: {summary['Total']} | Pass: {summary['Pass']} | Warn: {summary['Warnings']} | Fail: {summary['Fails']}</p>
</div>
{html_table}
</body></html>
"""
        )


def iter_audio_files(input_dir: str) -> Iterable[str]:
    """Yield audio files from the input directory (non-recursive)."""

    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(VALID_EXTENSIONS):
                yield entry.path


def process_directory(args: argparse.Namespace) -> None:
    """Process all audio files inside the given directory."""

    mode_config = MODES[args.mode]
    input_files = sorted(iter_audio_files(args.input))
    if not input_files:
        raise RuntimeError(f"No WAV or AIFF files found in '{args.input}'.")

    os.makedirs(args.output, exist_ok=True)
    reports_dir = os.path.join(args.output, "Reports")
    os.makedirs(reports_dir, exist_ok=True)

    analytics_data: List[Dict[str, object]] = []
    log_lines = [
        f"Starting batch process for folder: {args.input}",
        f"Mode: {mode_config['DESC']}",
    ]

    for file_path in input_files:
        basename = os.path.basename(file_path)
        log_lines.append(f"[INFO] Processing: {basename}")
        try:

            processed, sr, report = process_one_file(file_path, mode_config)

            base, ext = os.path.splitext(basename)
            safe_base = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in base)
            if args.suffix:
                suffix = (
                    f"__{args.mode}__-"
                    f"{abs(int(mode_config['TARGET_LKFS']))}LUFS_-"
                    f"{abs(int(mode_config['TP_LIMIT']))}dBTP"
                )
                out_filename = f"{safe_base}{suffix}{ext}"
            else:
                out_filename = f"prepped_{safe_base}{ext}"

            out_path = os.path.join(args.output, out_filename)
            seed = abs(hash(basename)) % (2**32)
            write_wav_24bit_tpdf(out_path, processed, sr, seed)

            fixed, before_tp, after_tp = enforce_tp_after_write(out_path, mode_config["TP_LIMIT"])
            if fixed:
                log_lines.append(
                    f"  -> [FIXED] Post-write peak was {before_tp:.2f} dBTP; corrected to {after_tp:.2f} dBTP."
                )
                report["status"] = "WARN (Peak Corrected)"
                report["post_tp_db"] = round(after_tp, 2)
            else:
                report["post_tp_db"] = round(after_tp, 2)

            analytics_data.append(report)
            log_lines.append(
                f"  -> Status: {report['status']}, Post LUFS: {report['post_lufs']}, Final Peak: {report['post_tp_db']} dBTP"
            )
        except ValueError as exc:
            log_lines.append(f"[SKIP] Skipped {basename}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            log_lines.append(f"[ERROR] Failed to process {basename}: {exc}")

    log_path = os.path.join(reports_dir, "processing_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("\n".join(log_lines))

    if not analytics_data:
        raise RuntimeError(f"No files were successfully processed. See log: {log_path}")

    df = pd.DataFrame(analytics_data)
    csv_path = os.path.join(reports_dir, "qc_report.csv")
    html_path = os.path.join(reports_dir, "summary_report.html")
    df.to_csv(csv_path, index=False)
    generate_html_report(df, html_path, mode_config["DESC"])

    print(
        f"\nBatch processing complete.\n"
        f"Processed files written to: {args.output}\n"
        f"Reports available at: {reports_dir}"
    )


def main() -> None:
    """Parse arguments and kick off directory processing."""

    parser = argparse.ArgumentParser(description="Prepare audio files for broadcast use.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_DIR,
        help=f"Path to the folder containing WAV/AIFF files (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination folder for processed audio files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--mode",
        choices=MODES.keys(),
        default="BG_PREP",
        help="Processing mode: BG_PREP or QC_MASTER.",
    )
    parser.add_argument(
        "--suffix",
        action="store_true",
        help="Append descriptive suffixes to output filenames.",
    )

    args = parser.parse_args()

    os.makedirs(args.input, exist_ok=True)

    process_directory(args)


if __name__ == "__main__":
    main()
