"""
GigaAM ASR backend using the GigaAM v3 e2e_rnnt model.

This backend provides Russian-optimized speech-to-text transcription using
GigaAM's Conformer-based acoustic model with RNN-T decoder. Supports both
short-form (<25s) and long-form transcription with VAD segmentation.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from whisperlivekit.timed_objects import ASRToken, Transcript

logger = logging.getLogger(__name__)

DEFAULT_GIGAAM_MODEL = "v3_e2e_rnnt"
GIGAAM_SAMPLE_RATE = 16000

# GigaAM short-form limit
GIGAAM_SHORTFORM_LIMIT_SECONDS = 25

# Streaming configuration
HOLDBACK_SECONDS = 1.5
MIN_NEW_SECONDS = 3.0
MAX_BUFFER_SECONDS = 45.0
TRIM_BEFORE_COMMITTED_SECONDS = 3


def _missing_dependency_error(reason: str) -> ImportError:
    return ImportError(
        "gigaam backend requires the GigaAM package. "
        "Install it with: pip install 'whisperlivekit[gigaam]'. "
        f"Details: {reason}"
    )


def _load_gigaam():
    """Load GigaAM package with dependency checking."""
    try:
        import gigaam
        import torch
        from gigaam import load_model
    except ImportError as exc:
        raise _missing_dependency_error(str(exc)) from exc
    return gigaam, torch, load_model


@dataclass
class _AlignedWord:
    """Word with timestamp information."""
    text: str
    start: float
    end: float


class GigaAMASR:
    """
    GigaAM ASR backend for Russian speech recognition.
    
    Uses the GigaAM v3 e2e_rnnt model which is optimized for Russian language
    with support for punctuation and text normalization.
    
    Features:
    - Short-form transcription (<25s): Direct inference
    - Long-form transcription: VAD-based segmentation with pyannote
    - Streaming: Buffer-based retranscription with holdback policy
    """
    
    sep = " "
    SAMPLING_RATE = GIGAAM_SAMPLE_RATE
    backend_choice = "gigaam"
    
    def __init__(self, logfile=sys.stderr, **kwargs):
        """
        Initialize GigaAM ASR backend.
        
        Args:
            logfile: Log file output
            **kwargs: Configuration options:
                - model_name: GigaAM model name (default: "v3_e2e_rnnt")
                - longform: Enable long-form transcription (default: False)
                - device: Device to use ("cuda", "cpu", or None for auto)
        """
        gigaam, self._torch, load_model = _load_gigaam()
        
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.model_name = kwargs.get("model_name", DEFAULT_GIGAAM_MODEL)
        self.longform = kwargs.get("longform", False)
        self.device = kwargs.get("device", None)
        
        # Load the model
        logger.info("Loading GigaAM model '%s' ...", self.model_name)
        try:
            self.model = load_model(self.model_name)
            if self.device:
                self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load GigaAM model {self.model_name}: {e}") from e
        
        logger.info("GigaAM model loaded successfully")
        
        # Warmup the model
        try:
            self.model.warmup(duration_seconds=1.0)
            logger.info("GigaAM model warmed up")
        except Exception as e:
            logger.warning("Model warmup failed: %s", e)
        
        self._device = self.model._device
        self._dtype = self.model._dtype
    
    def _convert_audio_to_wav(self, audio: np.ndarray, temp_path: str) -> None:
        """Convert numpy audio array to WAV file format."""
        import wave
        
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.SAMPLING_RATE)
            wav_file.writeframes(audio_int16.tobytes())
    
    def transcribe_text(self, audio: np.ndarray) -> Tuple[str, Optional[str]]:
        """
        Transcribe audio and return text with detected language.
        
        Args:
            audio: Audio array (16kHz mono float32)
            
        Returns:
            Tuple of (transcription_text, detected_language)
            Language is always "ru" for GigaAM as it's Russian-optimized
        """
        import tempfile
        import os
        
        if len(audio) < 320:  # Minimum audio length
            return "", None
        
        # Create temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            self._convert_audio_to_wav(audio, temp_path)
            # Use GigaAM's transcribe method
            result = self.model.transcribe(temp_path)
            # Ensure we get a string (handle any unexpected return types)
            if hasattr(result, 'text'):
                # It's a TranscriptionResult or similar object
                text = result.text if isinstance(result.text, str) else str(result.text)
            elif isinstance(result, str):
                text = result
            else:
                text = str(result)
            return text, "ru"  # GigaAM is Russian-optimized
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> str:
        """
        Transcribe audio (compatibility method).
        
        Args:
            audio: Audio array (16kHz mono float32)
            init_prompt: Ignored (not used by GigaAM)
            
        Returns:
            Transcription text
        """
        text, _ = self.transcribe_text(audio)
        return text
    
    def transcribe_longform(self, audio: np.ndarray) -> List[_AlignedWord]:
        """
        Transcribe long audio using VAD segmentation.
        
        Requires pyannote.audio for VAD. Falls back to short-form if not available.
        
        Args:
            audio: Audio array (16kHz mono float32)
            
        Returns:
            List of aligned words with timestamps
        """
        import tempfile
        import os
        
        if len(audio) < 320:
            return []
        
        # Create temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            self._convert_audio_to_wav(audio, temp_path)
            
            # Use longform transcription
            utterances = self.model.transcribe_longform(temp_path)
            
            # Convert to aligned words
            words = []
            for utterance in utterances:
                text = utterance.get("transcription", "")
                boundaries_raw = utterance.get("boundaries", (0.0, 0.0))
                # Ensure boundaries is a tuple of floats
                if isinstance(boundaries_raw, tuple) and len(boundaries_raw) == 2:
                    boundaries = (float(boundaries_raw[0]), float(boundaries_raw[1]))
                else:
                    boundaries = (0.0, 0.0)
                
                # Split text into words with approximate timestamps
                if not isinstance(text, str):
                    text = str(text)
                word_list = text.split()
                if word_list:
                    duration = boundaries[1] - boundaries[0]
                    word_duration = duration / len(word_list) if len(word_list) > 0 else 0
                    
                    for i, word in enumerate(word_list):
                        words.append(_AlignedWord(
                            text=word,
                            start=boundaries[0] + i * word_duration,
                            end=boundaries[0] + (i + 1) * word_duration,
                        ))
            
            return words
        except Exception as e:
            logger.warning("Longform transcription failed: %s", e)
            # Fall back to short-form
            text, _ = self.transcribe_text(audio)
            # Create a single word with full duration
            if text:
                return [_AlignedWord(text=text, start=0.0, end=len(audio) / self.SAMPLING_RATE)]
            return []
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def transcribe_aligned(self, audio: np.ndarray) -> Tuple[List[_AlignedWord], Optional[str]]:
        """
        Transcribe audio with word-level timestamps.
        
        Args:
            audio: Audio array (16kHz mono float32)
            
        Returns:
            Tuple of (aligned_words, detected_language)
        """
        # Use longform for better timestamps if enabled
        if self.longform:
            words = self.transcribe_longform(audio)
        else:
            # For short audio, use simple word splitting
            text, language = self.transcribe_text(audio)
            if not text:
                return [], language
            
            # Simple word-level timestamp estimation
            words = []
            word_list = text.split()
            if word_list:
                duration = len(audio) / self.SAMPLING_RATE
                word_duration = duration / len(word_list)
                
                for i, word in enumerate(word_list):
                    words.append(_AlignedWord(
                        text=word,
                        start=i * word_duration,
                        end=(i + 1) * word_duration,
                    ))
            
            return words, language
        
        language = "ru"  # GigaAM is Russian-optimized
        return words, language
    
    def use_vad(self) -> bool:
        """
        Check if VAD should be used.
        
        Returns:
            True if longform mode is enabled, False otherwise
        """
        return self.longform


class GigaAMOnlineProcessor:
    """
    Streaming processor for GigaAM ASR.
    
    Implements batch retranscription with holdback policy for streaming.
    Similar to Qwen3VLLMOnlineProcessor but adapted for GigaAM's architecture.
    """
    
    SAMPLING_RATE = GIGAAM_SAMPLE_RATE
    
    def __init__(self, asr: GigaAMASR, logfile=sys.stderr):
        """
        Initialize streaming processor.
        
        Args:
            asr: GigaAMASR instance
            logfile: Log file output
        """
        self.asr = asr
        self.logfile = logfile
        
        # Audio buffer
        self.end = 0.0
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer = []
        
        # Streaming state
        self._buffer_time_offset = 0.0
        self._last_committed_time: float = 0.0
        self._current_tokens: List[ASRToken] = []
        self._samples_since_last_inference = 0
        self._min_new_samples = int(MIN_NEW_SECONDS * self.SAMPLING_RATE)
    
    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        """
        Insert audio chunk into buffer.
        
        Args:
            audio: Audio chunk (16kHz mono float32)
            audio_stream_end_time: End time of audio in stream
        """
        self.end = audio_stream_end_time
        self.audio_buffer = np.append(self.audio_buffer, audio.astype(np.float32))
        self._samples_since_last_inference += len(audio)
    
    def _audio_duration(self) -> float:
        """Get current audio buffer duration in seconds."""
        return len(self.audio_buffer) / self.SAMPLING_RATE
    
    def _trim_buffer_if_needed(self):
        """Trim buffer if it exceeds maximum size."""
        duration = self._audio_duration()
        if duration <= MAX_BUFFER_SECONDS:
            return
        
        trim_to_time = self._last_committed_time - TRIM_BEFORE_COMMITTED_SECONDS
        if trim_to_time <= self._buffer_time_offset:
            return
        
        cut_samples = int((trim_to_time - self._buffer_time_offset) * self.SAMPLING_RATE)
        if cut_samples <= 0:
            return
        
        self.audio_buffer = self.audio_buffer[cut_samples:]
        self._buffer_time_offset += cut_samples / self.SAMPLING_RATE
        self._samples_since_last_inference = min(self._samples_since_last_inference, len(self.audio_buffer))
        self._current_tokens = []
    
    def _aligned_tokens(self) -> List[ASRToken]:
        """
        Get aligned tokens from current audio buffer.
        
        Returns:
            List of ASRToken with timestamps
        """
        if len(self.audio_buffer) < 320:
            return []
        
        aligned_words, detected_language = self.asr.transcribe_aligned(self.audio_buffer)
        tokens: List[ASRToken] = []
        
        for idx, word in enumerate(aligned_words):
            text = word.text if idx == 0 else " " + word.text
            tokens.append(
                ASRToken(
                    start=self._buffer_time_offset + word.start,
                    end=self._buffer_time_offset + word.end,
                    text=text,
                    detected_language=detected_language,
                )
            )
        
        self._current_tokens = tokens
        return tokens
    
    def _commit_available(self, flush: bool = False) -> List[ASRToken]:
        """
        Commit available tokens based on holdback policy.
        
        Args:
            flush: If True, commit all tokens (no holdback)
            
        Returns:
            List of committed ASRToken
        """
        if len(self.audio_buffer) < 320:
            return []
        
        self._trim_buffer_if_needed()
        tokens = self._aligned_tokens()
        
        if not tokens:
            return []
        
        # Calculate cutoff based on holdback
        cutoff = (
            self._buffer_time_offset + self._audio_duration()
            if flush
            else self._buffer_time_offset + self._audio_duration() - HOLDBACK_SECONDS
        )
        
        # Find tokens to commit
        start_idx = 0
        while start_idx < len(tokens):
            end_time = tokens[start_idx].end
            if end_time is not None and end_time > self._last_committed_time + 0.05:
                break
            start_idx += 1
        
        end_idx = start_idx
        while end_idx < len(tokens):
            end_time = tokens[end_idx].end
            if end_time is not None and end_time > cutoff:
                break
            end_idx += 1
        
        committed = tokens[start_idx:end_idx]
        if committed and committed[-1].end is not None:
            self._last_committed_time = committed[-1].end
        
        return committed
    
    def process_iter(self, is_last: bool = False) -> Tuple[List[ASRToken], float]:
        """
        Process current audio and return available tokens.
        
        Args:
            is_last: If True, this is the last chunk
            
        Returns:
            Tuple of (committed_tokens, end_time)
        """
        try:
            # Skip if not enough new audio
            if not is_last and self._samples_since_last_inference < self._min_new_samples:
                return [], self.end
            
            self._samples_since_last_inference = 0
            return self._commit_available(flush=is_last), self.end
        except Exception as e:
            logger.warning("[gigaam] process_iter error: %s", e, exc_info=True)
            return [], self.end
    
    def get_buffer(self) -> Transcript:
        """
        Get current buffer transcript (uncommitted tokens).
        
        Returns:
            Transcript of uncommitted tokens
        """
        tokens = [
            token
            for token in self._current_tokens
            if token.end is not None and token.end > self._last_committed_time + 0.05
        ]
        return Transcript.from_tokens(tokens=tokens, sep=self.asr.sep)
    
    def _reset_for_next_utterance(self):
        """Reset state for next utterance."""
        self._buffer_time_offset += self._audio_duration()
        self._last_committed_time = self._buffer_time_offset
        self.audio_buffer = np.array([], dtype=np.float32)
        self._samples_since_last_inference = 0
        self._current_tokens = []
    
    def start_silence(self) -> Tuple[List[ASRToken], float]:
        """
        Handle start of silence (utterance boundary).
        
        Args:
            silence_start: Start time of silence
            
        Returns:
            Tuple of (flushed_tokens, end_time)
        """
        tokens = self._commit_available(flush=True)
        logger.info("[gigaam] start_silence: flushed %d words", len(tokens))
        self._reset_for_next_utterance()
        return tokens, self.end
    
    def end_silence(self, silence_duration: float, offset: float):
        """
        Handle end of silence.
        
        Args:
            silence_duration: Duration of silence in seconds
            offset: Offset time
        """
        self._buffer_time_offset += silence_duration
        self._last_committed_time += silence_duration
        self.end += silence_duration
    
    def new_speaker(self, change_speaker):
        """
        Handle new speaker detection.
        
        Args:
            change_speaker: Speaker change information
        """
        self.start_silence()
    
    def warmup(self, audio: np.ndarray, init_prompt: str = "") -> Optional[ASRToken]:
        """
        Warmup the model with sample audio.
        
        Args:
            audio: Sample audio for warmup
            init_prompt: Ignored
            
        Returns:
            None
        """
        # Model already warmed up during initialization
        return None
    
    def finish(self) -> Tuple[List[ASRToken], float]:
        """
        Finish transcription and flush remaining tokens.
        
        Returns:
            Tuple of (final_tokens, end_time)
        """
        tokens = self._commit_available(flush=True)
        logger.info("[gigaam] finish: flushed %d words", len(tokens))
        return tokens, self.end
