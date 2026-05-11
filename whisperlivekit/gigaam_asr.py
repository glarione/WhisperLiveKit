"""GigaAM model backend for WhisperLiveKit."""

import tempfile
from typing import Dict, List

import gigaam
import numpy as np
from loguru import logger

from whisperlivekit.local_agreement.backends import ASRBase, ASRToken


class GigaAMASR(ASRBase):
    """
    GigaAM model backend for WhisperLiveKit streaming transcription.

    This backend integrates GigaAM's ASR models (v1/v2/v3) with the
    WhisperLiveKit streaming pipeline.

    Supports:
    - v3_e2e_rnnt (recommended for best quality)
    """

    sep = " "  # Space separator for text output

    def __init__(self, *args, **kwargs):
        """Initialize GigaAM ASR backend."""
        super().__init__(*args, **kwargs)
        self.model = None
        self.model_name = kwargs.get("model_name", "v3_e2e_rnnt")
        self.sample_rate = 16000  # GigaAM expects 16kHz audio

    def load_model(self, model_path: str = None, **kwargs):
        """
        Load GigaAM model.

        Args:
            model_path: Model name (e.g., 'v3_e2e_rnnt', 'v3_ctc')
            **kwargs: Additional model loading parameters

        Returns:
            Loaded GigaAM model
        """
        model_name = model_path or self.model_name
        logger.info(f"Loading GigaAM model: {model_name}")

        try:
            self.model = gigaam.load_model(model_name)
            logger.info(f"GigaAM model loaded successfully: {model_name}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load GigaAM model {model_name}: {e}")
            raise

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> Dict:
        """
        Transcribe audio chunk using GigaAM model.

        Args:
            audio: np.ndarray, shape (N_samples,), float32, 16kHz, normalized [-1, 1]
            init_prompt: Optional context prompt (currently not used by GigaAM)

        Returns:
            Dict with 'segments' list containing word-level timestamps:
            {
                'segments': [
                    {
                        'start': 0.5,      # seconds
                        'end': 1.2,        # seconds
                        'words': [
                            {
                                'start': 0.5,
                                'end': 0.7,
                                'word': 'hello',
                                'probability': 0.95
                            },
                            ...
                        ]
                    },
                    ...
                ]
            }
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if audio is None or len(audio) == 0:
            return {"segments": []}

        # Convert float32 [-1, 1] to int16 for GigaAM
        # GigaAM expects int16 PCM data
        audio_int16 = (audio * 32767).astype(np.int16)

        # Save to temporary WAV file (GigaAM API expects file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            import wave

            # Write WAV file
            with wave.open(tmp_file.name, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes (int16)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            tmp_path = tmp_file.name

        try:
            # Transcribe with word-level timestamps
            logger.debug(f"Transcribing {len(audio) / self.sample_rate:.2f}s of audio...")

            result = self.model.transcribe(tmp_path, word_timestamps=True)

            # Convert GigaAM result format to WhisperLiveKit format
            return self._format_result(result)

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            import traceback

            traceback.print_exc()
            return {"segments": []}

        finally:
            # Clean up temporary file
            import os

            try:
                os.unlink(tmp_path)
            except:
                pass

    def _format_result(self, gigaam_result) -> Dict:
        """
        Convert GigaAM result format to WhisperLiveKit format.

        GigaAM result structure:
            result.words: List of Word objects with:
                - start: float (seconds)
                - end: float (seconds)
                - text: str
                - confidence: float (optional)

        Returns WhisperLiveKit format with segments and words.
        """
        segments = []

        if not hasattr(gigaam_result, "words") or not gigaam_result.words:
            return {"segments": []}

        # Group words into segments (simple: one segment per continuous speech)
        current_segment = {"start": gigaam_result.words[0].start, "end": gigaam_result.words[0].end, "words": []}

        for word in gigaam_result.words:
            # Check if word is continuous with current segment
            if abs(word.start - current_segment["end"]) < 0.5:  # 0.5s threshold
                # Add to current segment
                current_segment["words"].append(
                    {
                        "start": float(word.start),
                        "end": float(word.end),
                        "word": word.text,
                        "probability": getattr(word, "confidence", None),
                    }
                )
                current_segment["end"] = float(word.end)
            else:
                # Start new segment
                if current_segment["words"]:
                    segments.append(current_segment)

                current_segment = {
                    "start": float(word.start),
                    "end": float(word.end),
                    "words": [
                        {
                            "start": float(word.start),
                            "end": float(word.end),
                            "word": word.text,
                            "probability": getattr(word, "confidence", None),
                        }
                    ],
                }

        # Don't forget the last segment
        if current_segment["words"]:
            segments.append(current_segment)

        return {"segments": segments}

    def ts_words(self, result: Dict) -> List[ASRToken]:
        """
        Extract word-level tokens from transcription result.

        Args:
            result: Dict from transcribe() method

        Returns:
            List[ASRToken] with fields:
                - start: float (seconds)
                - end: float (seconds)
                - text: str
                - probability: float (optional)
        """
        tokens = []

        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                tokens.append(
                    ASRToken(
                        start=word["start"], end=word["end"], text=word["word"], probability=word.get("probability")
                    )
                )

        return tokens

    def segments_end_ts(self, result: Dict) -> List[float]:
        """
        Return segment end timestamps for buffer trimming.

        Args:
            result: Dict from transcribe() method

        Returns:
            List[float] of segment end timestamps
        """
        return [segment["end"] for segment in result.get("segments", [])]

    def transcribe_longform(self, audio: np.ndarray) -> List[Dict]:
        """
        Transcribe long-form audio using GigaAM's longform transcription.

        Args:
            audio: np.ndarray, shape (N_samples,), float32, 16kHz

        Returns:
            List of segment dicts with text, start, end
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert float32 to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            import wave

            with wave.open(tmp_file.name, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            tmp_path = tmp_file.name

        try:
            # Use GigaAM's longform transcription
            result = self.model.transcribe_longform(tmp_path)

            # Convert to list of segments
            segments = []
            for segment in result:
                segments.append({"start": float(segment.start), "end": float(segment.end), "text": segment.text})

            return segments

        except Exception as e:
            logger.error(f"Longform transcription error: {e}")
            import traceback

            traceback.print_exc()
            return []

        finally:
            # Clean up
            import os

            try:
                os.unlink(tmp_path)
            except:
                pass
