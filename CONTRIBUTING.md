# Contributing to Backtrack-Remaster

Thank you for your interest in contributing to the Broadcast Audio Prep Tool!

## Getting Started

1. **Fork the repository** and clone your fork locally
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
3. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```

## Development Setup

### Prerequisites

- Python 3.8+
- libsndfile (platform-specific)
- FFmpeg (optional, for extended format support)

### Installing for Development

```bash
git clone https://github.com/YOUR_USERNAME/Backtrack-Remaster.git
cd Backtrack-Remaster
pip install -e ".[dev]"
```

## Making Changes

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Keep functions focused and single-purpose
- Document complex algorithms with inline comments

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add parallel processing with --jobs flag
fix: correct true-peak calculation for mono files
docs: update installation instructions for macOS
test: add unit tests for dithering function
```

Prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Adding or updating tests
- `refactor:` - Code change that neither fixes a bug nor adds a feature
- `perf:` - Performance improvement

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure all tests pass: `pytest`
4. Update documentation if needed
5. Submit a PR with a clear description

## Adding New Processing Modes

To add a custom processing mode:

1. Add the mode configuration to the `MODES` dictionary in `prep_audio.py`
2. Include target LUFS, true-peak limit, limiter ceiling, and description
3. Add tests for the new mode
4. Update documentation

Example:
```python
MODES["PODCAST"] = {
    "TARGET_LKFS": -16.0,
    "TP_LIMIT": -1.0,
    "LIM_CEIL": -2.0,
    "DESC": "Podcast Master (-16 LUFS / -1 dBTP)",
}
```

## Reporting Issues

When reporting bugs, please include:

- Python version (`python --version`)
- Operating system
- Full error traceback
- Sample audio file details (format, channels, sample rate)
- Command used

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions?

Open an issue with the `question` label or start a discussion.

---

Thank you for contributing!
