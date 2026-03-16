"""Resemblyzer wrapper and speaker enrollment store.

Manages the ``VoiceEncoder`` from resemblyzer and maintains an in-memory
enrollment store mapping speaker names to their mean embedding vectors.
Thread-safe via ``threading.Lock``.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from voicetag.exceptions import EnrollmentError
from voicetag.models import SpeakerProfile
from voicetag.utils import load_audio


class SpeakerEncoder:
    """Resemblyzer-based speaker encoder with an enrollment store.

    Args:
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device: str = device
        self._encoder = None
        self._profiles: dict[str, SpeakerProfile] = {}
        self._lock: threading.Lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        """Lazy-load the resemblyzer VoiceEncoder on first use."""
        if self._encoder is None:
            logger.debug("Loading resemblyzer VoiceEncoder (device={})", self._device)
            import contextlib
            import io

            from resemblyzer import VoiceEncoder as _VoiceEncoder

            # Suppress resemblyzer's "Loaded the voice encoder model..." print
            with contextlib.redirect_stdout(io.StringIO()):
                self._encoder = _VoiceEncoder(self._device)
            logger.info("Resemblyzer VoiceEncoder loaded successfully")

    def enroll(
        self,
        name: str,
        audio_paths: list[str | Path],
    ) -> SpeakerProfile:
        """Enroll a speaker by computing the mean embedding from audio files.

        Args:
            name: Speaker name to register.
            audio_paths: One or more paths to audio files of the speaker.

        Returns:
            The created ``SpeakerProfile``.

        Raises:
            EnrollmentError: If no valid audio files are provided or
                embedding computation fails.
        """
        if not audio_paths:
            raise EnrollmentError(
                f"Cannot enroll '{name}': no audio files provided."
            )

        self._ensure_loaded()
        embeddings: list[np.ndarray] = []

        for audio_path in audio_paths:
            try:
                audio, sr = load_audio(audio_path)
                emb = self.get_embedding(audio, sr)
                embeddings.append(emb)
            except Exception as exc:
                logger.warning(
                    "Skipping '{}' during enrollment of '{}': {}",
                    audio_path,
                    name,
                    exc,
                )

        if not embeddings:
            raise EnrollmentError(
                f"Cannot enroll '{name}': no valid audio files provided."
            )

        mean_embedding = np.mean(embeddings, axis=0)
        mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-10)

        profile = SpeakerProfile(
            name=name,
            embedding=mean_embedding.tolist(),
            num_samples=len(embeddings),
        )

        with self._lock:
            self._profiles[name] = profile

        logger.info(
            "Enrolled speaker '{}' from {} audio file(s)", name, len(embeddings)
        )
        return profile

    def get_embedding(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Compute a speaker embedding for an audio waveform.

        Args:
            audio: 1-D float32 waveform.
            sr: Sample rate (should be 16 000 for resemblyzer).

        Returns:
            256-dimensional numpy embedding vector.
        """
        self._ensure_loaded()
        from resemblyzer import preprocess_wav

        wav = preprocess_wav(audio, source_sr=sr)
        embedding = self._encoder.embed_utterance(wav)
        return embedding

    def compare(
        self,
        embedding: np.ndarray,
        profiles: Optional[dict[str, SpeakerProfile]] = None,
    ) -> tuple[str, float]:
        """Find the best matching speaker for an embedding.

        Args:
            embedding: 256-dimensional embedding vector to match.
            profiles: Optional external profile dict. If ``None``, uses the
                internal enrollment store.

        Returns:
            Tuple of ``(speaker_name, similarity_score)``. If no profiles
            are available, returns ``("UNKNOWN", 0.0)``.
        """
        with self._lock:
            search_profiles = profiles if profiles is not None else dict(self._profiles)

        if not search_profiles:
            return ("UNKNOWN", 0.0)

        best_name = "UNKNOWN"
        best_score = -1.0

        for name, profile in search_profiles.items():
            profile_embedding = np.array(profile.embedding, dtype=np.float32)
            score = self._cosine_similarity(embedding, profile_embedding)
            if score > best_score:
                best_score = score
                best_name = name

        return (best_name, float(best_score))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity in [-1, 1].
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def save_profiles(self, path: str | Path) -> None:
        """Save enrolled speaker profiles to a JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        with self._lock:
            profiles_data = {
                name: profile.model_dump(mode="json")
                for name, profile in self._profiles.items()
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(profiles_data, f, indent=2, default=str)

        logger.info("Saved {} profile(s) to {}", len(profiles_data), path)

    def load_profiles(self, path: str | Path) -> None:
        """Load speaker profiles from a JSON file.

        Args:
            path: Path to the profiles JSON file.

        Raises:
            EnrollmentError: If the file cannot be read or parsed.
        """
        path = Path(path)
        if not path.exists():
            raise EnrollmentError(f"Profiles file not found: {path}")

        try:
            with open(path) as f:
                profiles_data = json.load(f)

            loaded: dict[str, SpeakerProfile] = {}
            for name, data in profiles_data.items():
                loaded[name] = SpeakerProfile(**data)

            with self._lock:
                self._profiles.update(loaded)

            logger.info("Loaded {} profile(s) from {}", len(loaded), path)
        except Exception as exc:
            raise EnrollmentError(
                f"Failed to load profiles from '{path}': {exc}"
            ) from exc

    @property
    def enrolled_speakers(self) -> list[str]:
        """List of enrolled speaker names."""
        with self._lock:
            return list(self._profiles.keys())

    def remove_speaker(self, name: str) -> None:
        """Remove a speaker from the enrollment store.

        Args:
            name: Speaker name to remove.

        Raises:
            EnrollmentError: If the speaker is not enrolled.
        """
        with self._lock:
            if name not in self._profiles:
                raise EnrollmentError(
                    f"Speaker '{name}' is not enrolled. "
                    f"Enrolled speakers: {list(self._profiles.keys())}"
                )
            del self._profiles[name]

        logger.info("Removed speaker '{}'", name)
