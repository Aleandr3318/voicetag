"""Shared pytest fixtures for voicetag tests.

All fixtures are self-contained — no real ML models or audio hardware required.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicetag.models import SpeakerProfile, VoiceTagConfig


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_audio() -> np.ndarray:
    """Return a 5-second 16 kHz mono sine-wave (440 Hz) as float32."""
    sr = 16_000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture()
def sample_audio_file(sample_audio: np.ndarray, tmp_path: Path) -> Path:
    """Write *sample_audio* to a temporary WAV file and return its path."""
    wav_path = tmp_path / "test_audio.wav"
    sf.write(str(wav_path), sample_audio, 16_000)
    return wav_path


@pytest.fixture()
def sample_audio_files(tmp_path: Path) -> list[Path]:
    """Create 3 temp WAV files with slightly different sine-wave frequencies."""
    sr = 16_000
    duration = 3.0
    paths: list[Path] = []
    for i, freq in enumerate([440, 550, 660]):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        wav_path = tmp_path / f"test_audio_{i}.wav"
        sf.write(str(wav_path), audio, sr)
        paths.append(wav_path)
    return paths


# ---------------------------------------------------------------------------
# Mock data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_diarization_output() -> list[dict]:
    """Return a representative list of diarization segment dicts."""
    return [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.5},
        {"speaker": "SPEAKER_01", "start": 3.0, "end": 7.0},
        {"speaker": "SPEAKER_00", "start": 7.5, "end": 10.0},
    ]


@pytest.fixture()
def mock_speaker_profile() -> SpeakerProfile:
    """Return a SpeakerProfile with a random 256-d embedding."""
    rng = np.random.default_rng(42)
    embedding = rng.standard_normal(256).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return SpeakerProfile(
        name="alice",
        embedding=embedding.tolist(),
        num_samples=3,
        created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture()
def mock_profiles_dict() -> dict[str, SpeakerProfile]:
    """Return a dict with two speaker profiles (alice & bob)."""
    rng = np.random.default_rng(42)
    profiles: dict[str, SpeakerProfile] = {}
    for name in ("alice", "bob"):
        emb = rng.standard_normal(256).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        profiles[name] = SpeakerProfile(
            name=name,
            embedding=emb.tolist(),
            num_samples=2,
            created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )
    return profiles


# ---------------------------------------------------------------------------
# Config / path fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> VoiceTagConfig:
    """Return a VoiceTagConfig with safe defaults (no HF token needed)."""
    return VoiceTagConfig(
        hf_token="fake-token-for-tests",
        similarity_threshold=0.75,
        device="cpu",
    )


@pytest.fixture()
def tmp_profiles_path(tmp_path: Path) -> Path:
    """Return a temp file path suitable for profile save/load tests."""
    return tmp_path / "profiles.json"
