"""Tests for voicetag.encoder — SpeakerEncoder with mocked resemblyzer."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetag.encoder import SpeakerEncoder
from voicetag.exceptions import EnrollmentError
from voicetag.models import SpeakerProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embedding(seed: int = 0) -> np.ndarray:
    """Return a deterministic 256-d unit-norm embedding."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(256).astype(np.float32)
    return emb / np.linalg.norm(emb)


# We patch resemblyzer at the module-import level inside encoder.py.
# The encoder lazy-imports via _ensure_loaded and get_embedding.


@pytest.fixture()
def encoder():
    """Return a SpeakerEncoder with resemblyzer fully mocked."""
    enc = SpeakerEncoder(device="cpu")
    mock_voice_encoder = MagicMock()
    mock_voice_encoder.embed_utterance.return_value = _fake_embedding(0)
    enc._encoder = mock_voice_encoder  # skip lazy load
    return enc


# ---------------------------------------------------------------------------
# enroll
# ---------------------------------------------------------------------------


class TestEnroll:
    @patch("voicetag.encoder.load_audio")
    def test_enroll_adds_profile(
        self, mock_load, encoder: SpeakerEncoder, sample_audio_files: list[Path]
    ):
        mock_load.return_value = (np.zeros(16_000, dtype=np.float32), 16_000)
        with patch(
            "voicetag.encoder.SpeakerEncoder.get_embedding", return_value=_fake_embedding(1)
        ):
            profile = encoder.enroll("alice", [str(p) for p in sample_audio_files])
        assert isinstance(profile, SpeakerProfile)
        assert profile.name == "alice"
        assert profile.num_samples == 3
        assert "alice" in encoder.enrolled_speakers

    def test_enroll_empty_paths_raises(self, encoder: SpeakerEncoder):
        with pytest.raises(EnrollmentError, match="no audio files provided"):
            encoder.enroll("nobody", [])

    @patch("voicetag.encoder.load_audio")
    def test_enroll_all_files_fail_raises(self, mock_load, encoder: SpeakerEncoder):
        mock_load.side_effect = Exception("corrupt file")
        with pytest.raises(EnrollmentError, match="no valid audio files"):
            encoder.enroll("bad", ["/fake/a.wav", "/fake/b.wav"])


# ---------------------------------------------------------------------------
# get_embedding
# ---------------------------------------------------------------------------


class TestGetEmbedding:
    @patch("voicetag.encoder.preprocess_wav", create=True)
    def test_returns_numpy_array(self, mock_preprocess, encoder: SpeakerEncoder):
        # We need to patch the import inside get_embedding
        mock_preprocess.return_value = np.zeros(16_000, dtype=np.float32)
        with patch.dict("sys.modules", {"resemblyzer": MagicMock(preprocess_wav=mock_preprocess)}):
            emb = encoder.get_embedding(np.zeros(16_000, dtype=np.float32), 16_000)
        assert isinstance(emb, np.ndarray)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompare:
    def test_finds_best_match(
        self, encoder: SpeakerEncoder, mock_profiles_dict: dict[str, SpeakerProfile]
    ):
        # Use alice's exact embedding so she scores highest
        alice_emb = np.array(mock_profiles_dict["alice"].embedding, dtype=np.float32)
        name, score = encoder.compare(alice_emb, profiles=mock_profiles_dict)
        assert name == "alice"
        assert score > 0.5

    def test_returns_unknown_when_no_profiles(self, encoder: SpeakerEncoder):
        name, score = encoder.compare(_fake_embedding(99), profiles={})
        assert name == "UNKNOWN"
        assert score == 0.0

    def test_returns_unknown_with_none_profiles_and_empty_store(self, encoder: SpeakerEncoder):
        name, score = encoder.compare(_fake_embedding(0))
        assert name == "UNKNOWN"
        assert score == 0.0


# ---------------------------------------------------------------------------
# save / load profiles
# ---------------------------------------------------------------------------


class TestProfilesPersistence:
    @patch("voicetag.encoder.load_audio")
    def test_save_load_roundtrip(
        self,
        mock_load,
        encoder: SpeakerEncoder,
        tmp_profiles_path: Path,
        sample_audio_files: list[Path],
    ):
        mock_load.return_value = (np.zeros(16_000, dtype=np.float32), 16_000)
        with patch(
            "voicetag.encoder.SpeakerEncoder.get_embedding", return_value=_fake_embedding(10)
        ):
            encoder.enroll("alice", [str(sample_audio_files[0])])

        encoder.save_profiles(tmp_profiles_path)
        assert tmp_profiles_path.exists()

        encoder2 = SpeakerEncoder()
        encoder2._encoder = MagicMock()  # skip lazy load
        encoder2.load_profiles(tmp_profiles_path)
        assert "alice" in encoder2.enrolled_speakers

    def test_load_missing_file_raises(self, encoder: SpeakerEncoder, tmp_path: Path):
        with pytest.raises(EnrollmentError, match="not found"):
            encoder.load_profiles(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# remove_speaker
# ---------------------------------------------------------------------------


class TestRemoveSpeaker:
    @patch("voicetag.encoder.load_audio")
    def test_remove_existing(
        self, mock_load, encoder: SpeakerEncoder, sample_audio_files: list[Path]
    ):
        mock_load.return_value = (np.zeros(16_000, dtype=np.float32), 16_000)
        with patch(
            "voicetag.encoder.SpeakerEncoder.get_embedding", return_value=_fake_embedding(5)
        ):
            encoder.enroll("temp_speaker", [str(sample_audio_files[0])])
        assert "temp_speaker" in encoder.enrolled_speakers

        encoder.remove_speaker("temp_speaker")
        assert "temp_speaker" not in encoder.enrolled_speakers

    def test_remove_missing_raises(self, encoder: SpeakerEncoder):
        with pytest.raises(EnrollmentError, match="not enrolled"):
            encoder.remove_speaker("ghost")


# ---------------------------------------------------------------------------
# enrolled_speakers property
# ---------------------------------------------------------------------------


class TestEnrolledSpeakers:
    def test_initially_empty(self, encoder: SpeakerEncoder):
        assert encoder.enrolled_speakers == []

    @patch("voicetag.encoder.load_audio")
    def test_reflects_enrollment(
        self, mock_load, encoder: SpeakerEncoder, sample_audio_files: list[Path]
    ):
        mock_load.return_value = (np.zeros(16_000, dtype=np.float32), 16_000)
        with patch(
            "voicetag.encoder.SpeakerEncoder.get_embedding", return_value=_fake_embedding(0)
        ):
            encoder.enroll("speaker_a", [str(sample_audio_files[0])])
            encoder.enroll("speaker_b", [str(sample_audio_files[1])])
        assert set(encoder.enrolled_speakers) == {"speaker_a", "speaker_b"}


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    @patch("voicetag.encoder.load_audio")
    def test_concurrent_enroll(
        self, mock_load, encoder: SpeakerEncoder, sample_audio_files: list[Path]
    ):
        mock_load.return_value = (np.zeros(16_000, dtype=np.float32), 16_000)

        errors: list[Exception] = []

        def _enroll(name: str) -> None:
            try:
                with patch(
                    "voicetag.encoder.SpeakerEncoder.get_embedding",
                    return_value=_fake_embedding(hash(name) % 100),
                ):
                    encoder.enroll(name, [str(sample_audio_files[0])])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_enroll, args=(f"speaker_{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(encoder.enrolled_speakers) == 10
