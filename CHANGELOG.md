# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-16

### Added
- Initial release
- Core speaker diarization pipeline using pyannote.audio
- Speaker identification using resemblyzer embeddings
- Overlap detection for simultaneous speech regions
- CLI with `enroll`, `identify`, and `profiles` commands
- Save/load speaker profiles to JSON
- Pydantic v2 result models (`DiarizationResult`, `SpeakerSegment`, `OverlapSegment`, `SpeakerProfile`)
- `VoiceTagConfig` with environment-aware HuggingFace token resolution
- Parallel embedding computation via `ThreadPoolExecutor`
- Custom exception hierarchy (`VoiceTagError` and subclasses)
- Comprehensive test suite (77 tests, 83% coverage)
