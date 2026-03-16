"""Pyannote.audio wrapper for speaker diarization.

Handles HuggingFace authentication, lazy model loading, and conversion
of pyannote ``Annotation`` objects into plain dicts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from voicetag.exceptions import DiarizationError, VoiceTagConfigError


class Diarizer:
    """Speaker diarization via pyannote.audio.

    Args:
        hf_token: HuggingFace API token. Falls back to the ``HF_TOKEN``
            environment variable if not provided.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self._hf_token: Optional[str] = hf_token or os.environ.get("HF_TOKEN")
        self._device: str = device
        self._pipeline: Any = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the pyannote diarization pipeline on first use.

        Raises:
            VoiceTagConfigError: If no HuggingFace token is available.
            DiarizationError: If the pipeline fails to load.
        """
        if self._pipeline is not None:
            return

        if not self._hf_token:
            raise VoiceTagConfigError(
                "HuggingFace token required for pyannote diarization. "
                "Set hf_token in VoiceTagConfig, export the HF_TOKEN "
                "environment variable, or create a token at "
                "https://huggingface.co/settings/tokens"
            )

        logger.debug("Loading pyannote diarization pipeline (device={})", self._device)

        try:
            from pyannote.audio import Pipeline as PyannotePipeline

            self._pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self._hf_token,
            )

            import torch

            device = torch.device(self._device)
            self._pipeline.to(device)

            logger.info("Pyannote diarization pipeline loaded successfully")
        except Exception as exc:
            error_msg = str(exc)
            if "401" in error_msg or "authentication" in error_msg.lower():
                raise DiarizationError(
                    "Diarization failed: model requires authentication (HTTP 401). "
                    "Ensure your HuggingFace token is valid and you have accepted "
                    "the model license at "
                    "https://huggingface.co/pyannote/speaker-diarization-3.1"
                ) from exc
            raise DiarizationError(
                f"Failed to load pyannote diarization pipeline: {exc}"
            ) from exc

    def diarize(self, audio_path: str | Path) -> list[dict]:
        """Run speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of segment dicts with keys ``"speaker"``, ``"start"``,
            and ``"end"``.

        Raises:
            DiarizationError: If the diarization pipeline fails.
        """
        self._ensure_loaded()
        audio_path = Path(audio_path).resolve()
        logger.debug("Running diarization on {}", audio_path.name)

        try:
            import warnings

            # Suppress pyannote TF32 and std warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                annotation = self._pipeline(str(audio_path))
            segments = self._parse_annotation(annotation)
            logger.info(
                "Diarization complete: {} segments detected", len(segments)
            )
            return segments
        except Exception as exc:
            raise DiarizationError(
                f"Diarization failed for '{audio_path.name}': {exc}"
            ) from exc

    @staticmethod
    def _parse_annotation(annotation: Any) -> list[dict]:
        """Convert a pyannote ``Annotation`` into a list of segment dicts.

        Args:
            annotation: A ``pyannote.core.Annotation`` object.

        Returns:
            List of dicts, each with ``"speaker"`` (str), ``"start"``
            (float), and ``"end"`` (float).
        """
        segments: list[dict] = []

        # pyannote >=3.3 returns DiarizeOutput with .speaker_diarization attribute
        # pyannote <3.3 returns Annotation directly
        ann = annotation
        if hasattr(annotation, "speaker_diarization"):
            ann = annotation.speaker_diarization
        elif hasattr(annotation, "itertracks"):
            ann = annotation

        for turn, _, speaker in ann.itertracks(yield_label=True):
            segments.append(
                {
                    "speaker": str(speaker),
                    "start": float(turn.start),
                    "end": float(turn.end),
                }
            )
        return segments
