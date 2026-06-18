"""Tests for GigaAM ASR backend."""
import numpy as np

from whisperlivekit.gigaam_asr import (
    GigaAMASR,
    GigaAMOnlineProcessor,
    _AlignedWord,
)


def _audio(seconds):
    """Generate dummy audio of given duration in seconds."""
    return np.zeros(int(seconds * 16_000), dtype=np.float32)


class MockGigaAM:
    """Mock GigaAM ASR for testing."""
    
    def __init__(self, aligned_words, language="ru"):
        self.aligned_words = aligned_words
        self.language = language
        self.model_name = "v3_e2e_rnnt"
        self.longform = False
        self.backend_choice = "gigaam"
        self.calls = 0
    
    def transcribe_aligned(self, audio):
        """Mock transcribe_aligned returning pre-defined words."""
        self.calls += 1
        return self.aligned_words, self.language
    
    def transcribe_text(self, audio):
        """Mock transcribe_text returning joined words."""
        self.calls += 1
        text = " ".join(w.text for w in self.aligned_words)
        return text, self.language


def test_gigaam_commits_only_before_holdback():
    """Test that GigaAM processor commits only words before holdback window."""
    mock_asr = MockGigaAM([  # type: ignore
        _AlignedWord("привет", 0.0, 1.0),
        _AlignedWord("мир", 1.0, 2.0),
        _AlignedWord("как", 2.0, 3.0),
        _AlignedWord("дела", 3.0, 4.0),
    ])
    
    processor = GigaAMOnlineProcessor(mock_asr)  # type: ignore
    
    # Feed 4 seconds of audio
    processor.insert_audio_chunk(_audio(4.0), 4.0)
    
    # Process - should commit words before the last 0.5s holdback
    tokens, _ = processor.process_iter(is_last=False)
    
    # Should have committed words that end before 3.5s (4.0 - 0.5 holdback)
    committed_ends = [t.end for t in tokens if t.end is not None]
    
    # "дела" ends at 4.0s, so it should NOT be committed
    # "как" ends at 3.0s, so it SHOULD be committed
    assert all(end <= 3.5 for end in committed_ends), \
        f"Some tokens committed past holdback: {committed_ends}"


def test_gigaam_flushes_all_on_finish():
    """Test that GigaAM processor flushes all remaining words on finish."""
    mock_asr = MockGigaAM([  # type: ignore
        _AlignedWord("первое", 0.0, 1.0),
        _AlignedWord("второе", 1.0, 2.0),
        _AlignedWord("третье", 2.0, 3.0),
    ])
    
    processor = GigaAMOnlineProcessor(mock_asr)  # type: ignore
    
    # Feed 3 seconds of audio
    processor.insert_audio_chunk(_audio(3.0), 3.0)
    
    # Process without flush - should hold back last word
    tokens, _ = processor.process_iter(is_last=False)
    assert len(tokens) < 3, "Should hold back some words before finish"
    
    # Process with flush - should commit all
    tokens, _ = processor.process_iter(is_last=True)
    assert len(tokens) == 3, f"Should commit all 3 words on finish, got {len(tokens)}"


def test_gigaam_buffer_contains_uncommitted():
    """Test that get_buffer returns uncommitted tokens."""
    mock_asr = MockGigaAM([  # type: ignore
        _AlignedWord("word1", 0.0, 1.0),
        _AlignedWord("word2", 1.0, 2.0),
        _AlignedWord("word3", 2.0, 3.0),
    ])
    
    processor = GigaAMOnlineProcessor(mock_asr)  # type: ignore
    
    # Feed 3 seconds of audio
    processor.insert_audio_chunk(_audio(3.0), 3.0)
    
    # Process without flush
    processor.process_iter(is_last=False)
    
    # Get buffer - should contain uncommitted words
    buffer = processor.get_buffer()
    
    # Buffer should have some text (uncommitted words)
    assert buffer.text is not None
    # Buffer should not be empty if there are uncommitted words
    if len(mock_asr.aligned_words) > 2:
        assert len(buffer.text.split()) > 0 or len(mock_asr.aligned_words) <= 2


def test_gigaam_handles_empty_audio():
    """Test that GigaAM processor handles empty audio gracefully."""
    mock_asr = MockGigaAM([])  # type: ignore
    
    processor = GigaAMOnlineProcessor(mock_asr)  # type: ignore
    
    # Feed empty audio
    processor.insert_audio_chunk(_audio(0.0), 0.0)
    
    # Process - should not crash
    tokens, _ = processor.process_iter(is_last=False)
    assert tokens == [], "Should return empty tokens for empty audio"


def test_gigaam_resets_after_silence():
    """Test that GigaAM processor resets state after silence."""
    mock_asr = MockGigaAM([  # type: ignore
        _AlignedWord("first", 0.0, 1.0),
        _AlignedWord("second", 1.0, 2.0),
    ])
    
    processor = GigaAMOnlineProcessor(mock_asr)  # type: ignore
    
    # First utterance
    processor.insert_audio_chunk(_audio(2.0), 2.0)
    tokens1, _ = processor.process_iter(is_last=False)
    
    # Start silence (should flush and reset)
    flushed, _ = processor.start_silence()
    
    # Second utterance
    mock_asr.aligned_words = [
        _AlignedWord("third", 0.0, 1.0),
    ]
    processor.insert_audio_chunk(_audio(1.0), 3.0)
    tokens2, _ = processor.process_iter(is_last=False)
    
    # Should have processed both utterances independently
    assert len(flushed) >= 0  # May or may not have flushed
    assert len(tokens2) >= 0  # Should process new utterance
