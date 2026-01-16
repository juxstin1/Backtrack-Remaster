# Technical Documentation

This document provides in-depth technical details about the audio processing algorithms and standards used in the Broadcast Audio Prep Tool.

## Audio Processing Pipeline

The tool processes audio through the following stages:

```
Input File → DC Offset Removal → Resampling → Loudness Measurement
     → Gain Adjustment → Peak Limiting → Dithering → Output File
```

### 1. DC Offset Removal

Before any processing, DC offset is removed from each channel independently:

```python
data = data - np.mean(data, axis=0, keepdims=True)
```

This ensures the audio signal is centered around zero, which is important for accurate loudness measurement and preventing asymmetric clipping.

### 2. Resampling

All audio is resampled to 48 kHz using a high-quality polyphase filter:

```python
data = resample_poly(data, TARGET_SR, sr, axis=0, window=("kaiser", 7.5))
```

The Kaiser window with β=7.5 provides excellent stopband attenuation (~80 dB) while maintaining good passband flatness.

### 3. Loudness Measurement (ITU-R BS.1770)

Loudness is measured according to ITU-R BS.1770 using the `pyloudnorm` library:

- **K-weighting**: Pre-filter that models human perception
- **Mean square calculation**: Per-channel power measurement
- **Channel summing**: With appropriate weighting (1.0 for L/R, 1.41 for surround)
- **Gating**: -70 LUFS absolute gate, -10 LU relative gate

The result is expressed in LUFS (Loudness Units Full Scale).

### 4. Gain Adjustment

A uniform gain is applied to reach the target loudness:

```python
gain_db = TARGET_LKFS - measured_lufs
processed = audio * (10.0 ** (gain_db / 20.0))
```

### 5. True Peak Limiting

True peak measurement uses 4x oversampling to detect inter-sample peaks:

```python
oversampled = resample_poly(samples, 4, 1, axis=0, window=("kaiser", 7.5))
peak_linear = np.max(np.abs(oversampled))
```

A brick-wall limiter is applied if the signal exceeds the ceiling:

```python
if peak_linear > ceiling_linear:
    reduction_factor = ceiling_linear / peak_linear
    samples = samples * reduction_factor
```

### 6. TPDF Dithering

Triangular Probability Density Function (TPDF) dither is applied before quantization:

```python
dither = (rng.random(shape) - 0.5 + rng.random(shape) - 0.5) / (2**23)
```

Key properties:
- **Deterministic**: Seeded from filename for reproducibility
- **TPDF**: Sum of two uniform distributions creates triangular PDF
- **Amplitude**: 1 LSB of 24-bit audio

## Processing Modes

### BG_PREP (Background Music)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Target LUFS | -27 | Sits under dialogue without masking |
| True-Peak Limit | -6 dBTP | Headroom for mixing |
| Limiter Ceiling | -7 dB | 1 dB safety margin |

### QC_MASTER (Final Delivery)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Target LUFS | -24 | Broadcast standard |
| True-Peak Limit | -2 dBTP | EBU R128 / ATSC A/85 compliant |
| Limiter Ceiling | -3 dB | 1 dB safety margin |

## Post-Write Verification

After writing the file, the tool re-reads and measures the actual true peak. This is necessary because:

1. **Quantization effects**: Converting from float to 24-bit can introduce small errors
2. **Dithering**: Adds noise that may occasionally push peaks higher
3. **Encoding artifacts**: Some codecs may introduce minor level changes

If the true peak exceeds the limit, the file is scaled down and rewritten.

## Standards Compliance

### ITU-R BS.1770-4

The loudness measurement algorithm follows ITU-R BS.1770-4:
- K-frequency weighting
- Channel weighting
- Gated measurement

### EBU R128

The tool's output is compatible with EBU R128:
- Target loudness: -23 ±1 LUFS (QC_MASTER at -24 is within tolerance)
- True peak limit: ≤ -1 dBTP (QC_MASTER uses -2 dBTP)
- Loudness range: Not constrained

### ATSC A/85

For North American broadcast, ATSC A/85 specifies:
- Target loudness: -24 LKFS ±2 (QC_MASTER compliant)
- True peak limit: -2 dBTP (QC_MASTER compliant)

## File Format Support

### Input Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| WAV | .wav | All bit depths supported |
| AIFF | .aif, .aiff | Standard and extended formats |

### Output Format

All output files are:
- **Format**: WAV (RIFF)
- **Sample Rate**: 48 kHz
- **Bit Depth**: 24-bit PCM
- **Dithering**: TPDF

## Performance Considerations

### Memory Usage

- Files are loaded entirely into memory
- 4x oversampling for true peak measurement increases memory temporarily
- For a 1-hour stereo file at 48kHz: ~2.3 GB peak memory

### Parallelization

The `--jobs` flag enables parallel processing:
- Uses ThreadPoolExecutor for I/O-bound operations
- Each file is processed independently
- Recommended: 2-4 workers for typical systems

### Processing Speed

Typical processing times (Intel i7, 3.6 GHz):
- 3-minute song: ~2-3 seconds
- 1-hour podcast: ~30-45 seconds
- Batch of 100 files: ~5-10 minutes (sequential)

## References

- [ITU-R BS.1770-4](https://www.itu.int/rec/R-REC-BS.1770)
- [EBU R128](https://tech.ebu.ch/docs/r/r128.pdf)
- [ATSC A/85](https://www.atsc.org/standard/a85-techniques-for-establishing-and-maintaining-audio-loudness-for-digital-television/)
- [AES17-2015](https://www.aes.org/publications/standards/) - True peak measurement
