# Changelog

All notable changes to the Broadcast Audio Prep Tool will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Progress bar with tqdm for batch processing visibility
- `--version` flag to display current version
- `--dry-run` flag to preview processing without writing files
- `--verbose` and `--quiet` flags for output control
- `--recursive` flag to scan subdirectories
- `--jobs` flag for parallel processing
- JSON report output alongside CSV and HTML
- Comprehensive test suite with pytest
- CI/CD pipeline with GitHub Actions
- pyproject.toml for modern Python packaging

### Changed
- Improved logging with Python's logging module
- Enhanced error messages with full file paths

## [5.0.0] - 2024-01-15

### Changed
- Migrated from ZIP archive input to direct folder processing
- Simplified workflow: drop files in `input/`, run script, get results in `output/`

### Added
- HTML summary report with color-coded pass/fail status
- CSV export for spreadsheet analysis
- Processing log for debugging
- `--suffix` flag for descriptive output filenames

## [4.0.0] - 2023-XX-XX

### Added
- QC_MASTER mode for final delivery (-24 LUFS / -2 dBTP)
- Post-write true-peak verification and correction
- Deterministic TPDF dithering for reproducible output

### Changed
- Improved true-peak measurement with 4x oversampling

## [3.0.0] - 2023-XX-XX

### Added
- BG_PREP mode for background music (-27 LUFS / -6 dBTP)
- ITU-R BS.1770 loudness measurement via pyloudnorm
- High-quality resampling to 48 kHz
- 24-bit PCM output with TPDF dithering

### Changed
- Complete rewrite in Python
- Modular function design

## [2.0.0] - 2022-XX-XX

### Added
- Basic loudness normalization
- Peak limiting

## [1.0.0] - 2022-XX-XX

### Added
- Initial release
- Basic WAV file processing
