"""Tests for voicetag.utils — audio I/O utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voicetag.exceptions import AudioLoadError
from voicetag.utils import chunk_audio, load_audio, validate_audio_path


# ---------------------------------------------------------------------------
# validate_audio_path
# ---------------------------------------------------------------------------


class TestValidateAudioPath:
    def test_valid_wav(self, sample_audio_file: Path):
        result = validate_audio_path(sample_audio_file)
        assert result.exists()
        assert result.suffix == ".wav"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(AudioLoadError, match="not found"):
            validate_audio_path(tmp_path / "nonexistent.wav")

    def test_unsupported_format_raises(self, tmp_path: Path):
        txt_file = tmp_path / "audio.txt"
        txt_file.write_text("not audio")
        with pytest.raises(AudioLoadError, match="Unsupported audio format"):
            validate_audio_path(txt_file)

    def test_directory_raises(self, tmp_path: Path):
        with pytest.raises(AudioLoadError, match="not a file"):
            validate_audio_path(tmp_path)


# ---------------------------------------------------------------------------
# load_audio
# ---------------------------------------------------------------------------


class TestLoadAudio:
    def test_valid_wav(self, sample_audio_file: Path):
        audio, sr = load_audio(sample_audio_file)
        assert sr == 16_000
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        # 5 seconds at 16 kHz = 80 000 samples
        assert len(audio) == 80_000

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(AudioLoadError):
            load_audio(tmp_path / "missing.wav")

    def test_unsupported_format_raises(self, tmp_path: Path):
        bad_file = tmp_path / "audio.xyz"
        bad_file.write_text("garbage")
        with pytest.raises(AudioLoadError):
            load_audio(bad_file)


# ---------------------------------------------------------------------------
# chunk_audio
# ---------------------------------------------------------------------------


class TestChunkAudio:
    def test_correct_number_of_chunks(self, sample_audio: np.ndarray):
        sr = 16_000
        # 5s audio, 2s chunks, 0.5s overlap -> step=1.5s
        # Starts: 0, 1.5, 3.0, 4.5 -> 4 chunks
        chunks = chunk_audio(sample_audio, sr, chunk_duration=2.0, overlap=0.5)
        assert len(chunks) == 4
        # Each chunk is (array, offset)
        for chunk_arr, offset in chunks:
            assert isinstance(chunk_arr, np.ndarray)
            assert isinstance(offset, float)

    def test_short_audio_single_chunk(self):
        sr = 16_000
        short_audio = np.zeros(sr, dtype=np.float32)  # 1 second
        chunks = chunk_audio(short_audio, sr, chunk_duration=30.0, overlap=1.0)
        assert len(chunks) == 1
        assert chunks[0][1] == 0.0  # offset is 0

    def test_empty_audio(self):
        chunks = chunk_audio(np.array([], dtype=np.float32), 16_000)
        assert chunks == []

    def test_chunk_offsets_are_increasing(self, sample_audio: np.ndarray):
        chunks = chunk_audio(sample_audio, 16_000, chunk_duration=1.0, overlap=0.2)
        offsets = [offset for _, offset in chunks]
        assert offsets == sorted(offsets)
        # All offsets non-negative
        assert all(o >= 0 for o in offsets)
