"""voicetag — Speaker identification powered by pyannote and resemblyzer."""

from __future__ import annotations

from loguru import logger

logger.disable("voicetag")

from voicetag.exceptions import (
    AudioLoadError,
    DiarizationError,
    EnrollmentError,
    VoiceTagConfigError,
    VoiceTagError,
)
from voicetag.models import (
    DiarizationResult,
    OverlapSegment,
    SpeakerProfile,
    SpeakerSegment,
    VoiceTagConfig,
)
from voicetag.pipeline import Pipeline as VoiceTag

__version__ = "0.1.0"

__all__ = [
    "VoiceTag",
    "VoiceTagConfig",
    "SpeakerSegment",
    "OverlapSegment",
    "SpeakerProfile",
    "DiarizationResult",
    "VoiceTagError",
    "VoiceTagConfigError",
    "EnrollmentError",
    "DiarizationError",
    "AudioLoadError",
    "__version__",
]
