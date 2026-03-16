"""Audio I/O utilities for the voicetag library.

Provides functions for loading, validating, resampling, and chunking audio
files.  No ML dependencies — only ``soundfile`` and ``numpy``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

from voicetag.exceptions import AudioLoadError

SUPPORTED_EXTENSIONS: set[str] = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def validate_audio_path(path: str | Path) -> Path:
    """Validate that an audio file exists and has a supported format.

    Args:
        path: Path to the audio file.

    Returns:
        Resolved ``Path`` object.

    Raises:
        AudioLoadError: If the file does not exist or has an unsupported
            extension.
    """
    audio_path = Path(path).resolve()

    if not audio_path.exists():
        raise AudioLoadError(f"Audio file not found: {audio_path}")

    if not audio_path.is_file():
        raise AudioLoadError(f"Path is not a file: {audio_path}")

    suffix = audio_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise AudioLoadError(
            f"Unsupported audio format '{suffix}' for file '{audio_path.name}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    return audio_path


def load_audio(
    path: str | Path,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """Load an audio file, convert to mono, and resample to the target rate.

    Args:
        path: Path to the audio file.
        target_sr: Target sample rate in Hz. Defaults to 16 000.

    Returns:
        Tuple of ``(waveform, sample_rate)`` where *waveform* is a 1-D
        float32 numpy array and *sample_rate* equals ``target_sr``.

    Raises:
        AudioLoadError: If the file cannot be read or decoded.
    """
    audio_path = validate_audio_path(path)

    try:
        data, sr = sf.read(str(audio_path), dtype="float32")
    except Exception as exc:
        raise AudioLoadError(
            f"Cannot read audio file '{audio_path.name}': {exc}"
        ) from exc

    logger.debug("Loaded audio: {} samples, sr={}", len(data), sr)

    if data.ndim > 1:
        data = data.mean(axis=1)
        logger.debug("Converted to mono by averaging {} channels", data.shape[1] if data.ndim > 1 else 2)

    if sr != target_sr:
        data = _resample(data, sr, target_sr)
        logger.debug("Resampled from {} Hz to {} Hz", sr, target_sr)

    return data, target_sr


def _resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample a 1-D waveform using numpy linear interpolation.

    Args:
        audio: 1-D float32 waveform.
        orig_sr: Original sample rate.
        target_sr: Desired sample rate.

    Returns:
        Resampled 1-D float32 waveform.
    """
    if orig_sr == target_sr:
        return audio

    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    if target_length == 0:
        return np.array([], dtype=np.float32)

    indices = np.linspace(0, len(audio) - 1, target_length, dtype=np.float64)
    resampled = np.interp(indices, np.arange(len(audio), dtype=np.float64), audio)
    return resampled.astype(np.float32)


def chunk_audio(
    audio: np.ndarray,
    sr: int,
    chunk_duration: float = 30.0,
    overlap: float = 1.0,
) -> list[tuple[np.ndarray, float]]:
    """Split audio into overlapping chunks for memory-bounded processing.

    Args:
        audio: 1-D float32 waveform.
        sr: Sample rate of the waveform.
        chunk_duration: Duration of each chunk in seconds.
        overlap: Overlap between consecutive chunks in seconds.

    Returns:
        List of ``(chunk_array, start_offset)`` tuples where
        *start_offset* is the chunk's start time in seconds.
    """
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    step_samples = chunk_samples - overlap_samples

    if step_samples <= 0:
        step_samples = chunk_samples

    total_samples = len(audio)
    if total_samples == 0:
        return []

    chunks: list[tuple[np.ndarray, float]] = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        start_time = start / sr
        chunks.append((chunk, start_time))
        start += step_samples

    logger.debug(
        "Chunked {} samples into {} chunks (chunk={:.1f}s, overlap={:.1f}s)",
        total_samples,
        len(chunks),
        chunk_duration,
        overlap,
    )
    return chunks
