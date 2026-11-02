"""Regression coverage for silent-trim, silent-output, chunk-ceiling and
text-normalization fixes."""

import numpy as np
import pytest

from backend.backends import CHATTERBOX_MAX_CHUNK_CHARS, engine_max_chunk_chars
from backend.utils.audio import is_silent_audio, trim_tts_output
from backend.utils.chunked_tts import generate_chunked
from backend.utils.text_normalize import (
    normalize_for_tts,
    number_to_words,
    ordinal_to_words,
    year_to_words,
)

SR = 24000
_rng = np.random.default_rng(0)


def _speech(seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    env = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3 * t))
    return (_rng.standard_normal(len(t)) * amp * env).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


# --- trim_tts_output -------------------------------------------------------


def test_trim_keeps_normal_speech():
    audio = _speech(3, 0.15)
    out = trim_tts_output(audio, SR)
    assert len(out) / SR > 2.9


def test_trim_keeps_quiet_speech():
    # Real speech at ~-46 dBFS used to be trimmed to zero samples.
    audio = _speech(4, 0.006)
    out = trim_tts_output(audio, SR)
    assert len(out) / SR > 3.5


def test_trim_keeps_quiet_speech_after_click():
    audio = np.concatenate([_speech(0.04, 0.2), _speech(4, 0.006)])
    out = trim_tts_output(audio, SR)
    assert len(out) / SR > 3.5


def test_trim_never_empties_silent_input():
    audio = _silence(4)
    out = trim_tts_output(audio, SR)
    assert len(out) == len(audio)


def test_trim_still_cuts_post_silence_hallucination():
    audio = np.concatenate([_speech(2, 0.15), _silence(3), _speech(1, 0.15)])
    out = trim_tts_output(audio, SR)
    assert 1.9 < len(out) / SR < 2.5


def test_trim_cuts_hallucination_longer_than_the_speech():
    # Short real utterance, long gap, then a long hallucinated tail. The
    # never-empty guard must not mistake this for eating real speech.
    audio = np.concatenate([_speech(1, 0.15), _silence(3), _speech(2, 0.15)])
    out = trim_tts_output(audio, SR)
    assert 0.9 < len(out) / SR < 1.5


def test_is_silent_audio():
    assert is_silent_audio(np.array([], dtype=np.float32))
    assert is_silent_audio(_silence(1))
    assert not is_silent_audio(_speech(1, 0.006))


# --- generate_chunked silent-output guard and chunk ceiling ----------------


class _FakeBackend:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def generate(self, text, voice_prompt, language, seed, instruct):
        self.calls.append((text, seed))
        return self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0], SR


@pytest.mark.asyncio
async def test_silent_output_is_retried_once_then_succeeds():
    backend = _FakeBackend([_silence(1), _speech(1, 0.15)])
    audio, _ = await generate_chunked(backend, "Hello there.", {}, seed=5)
    assert not is_silent_audio(audio)
    assert len(backend.calls) == 2
    assert backend.calls[1][1] != 5


@pytest.mark.asyncio
async def test_silent_output_twice_raises():
    backend = _FakeBackend([_silence(1)])
    with pytest.raises(RuntimeError, match="silent"):
        await generate_chunked(backend, "Hello there.", {})


@pytest.mark.asyncio
async def test_trim_fn_returning_silence_keeps_untrimmed():
    backend = _FakeBackend([_speech(1, 0.15)])
    audio, _ = await generate_chunked(
        backend, "Hello there.", {}, trim_fn=lambda a, sr: np.zeros(0, dtype=np.float32)
    )
    assert len(audio) == SR


def test_chatterbox_engines_have_chunk_ceiling():
    assert engine_max_chunk_chars("chatterbox_turbo") == CHATTERBOX_MAX_CHUNK_CHARS
    assert engine_max_chunk_chars("chatterbox") == CHATTERBOX_MAX_CHUNK_CHARS
    assert engine_max_chunk_chars("kokoro") is None


@pytest.mark.asyncio
async def test_chunk_ceiling_caps_request_size():
    backend = _FakeBackend([_speech(0.5, 0.15)])
    text = " ".join(["This is a reasonably ordinary sentence of prose."] * 14)
    await generate_chunked(backend, text, {}, max_chunk_chars=800, max_chunk_ceiling=300)
    assert len(backend.calls) > 2
    assert all(len(t) <= 300 for t, _ in backend.calls)


@pytest.mark.asyncio
async def test_text_is_normalized_before_engine(monkeypatch):
    monkeypatch.delenv("VOICEBOX_TEXT_NORMALIZE", raising=False)
    backend = _FakeBackend([_speech(0.5, 0.15)])
    await generate_chunked(backend, "Up 37% today.", {})
    assert backend.calls[0][0] == "Up thirty-seven percent today."


@pytest.mark.asyncio
async def test_normalization_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VOICEBOX_TEXT_NORMALIZE", "0")
    backend = _FakeBackend([_speech(0.5, 0.15)])
    await generate_chunked(backend, "Up 37% today.", {})
    assert backend.calls[0][0] == "Up 37% today."


# --- text normalization ----------------------------------------------------


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "zero"),
        (42, "forty-two"),
        (1250, "one thousand two hundred fifty"),
        (17493, "seventeen thousand four hundred ninety-three"),
        (2_000_001, "two million one"),
    ],
)
def test_number_to_words(n, words):
    assert number_to_words(n) == words


def test_ordinals_and_years():
    assert [ordinal_to_words(n) for n in (1, 2, 3, 12, 20, 21)] == [
        "first", "second", "third", "twelfth", "twentieth", "twenty-first",
    ]
    assert year_to_words(1999) == "nineteen ninety-nine"
    assert year_to_words(2026) == "twenty twenty-six"
    assert year_to_words(2005) == "two thousand five"


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("$1,250.75", "one thousand two hundred fifty dollars and seventy-five cents"),
        ("on 11/02/2026,", "on November second, twenty twenty-six,"),
        ("2026-11-02", "November second, twenty twenty-six"),
        ("up 37%", "up thirty-seven percent"),
        ("3.5k users", "three point five thousand users"),
        ("ROI is 2:1", "ROI is two to one"),
        ("at 3:45 pm", "at three forty-five p m"),
        ("pages 5-10", "pages five to ten"),
        ("v2.0.1", "version two point oh point one"),
        ("72°F", "seventy-two degrees Fahrenheit"),
        ("audio.py#L150", "audio dot pie, line one hundred fifty"),
        ("trim_tts_output()", "trim T T S output"),
        ("git diff --stat && npm i", "git diff dash dash stat and then N P M i"),
        ("src/app.json", "source, app dot jason"),
        ("@types/node", "at types, node"),
        (
            "C:/Users/me/my-app/config.json,",
            "my app, config dot jason,",
        ),
        ("50/50", "fifty fifty"),
        ("and/or", "and or"),
    ],
)
def test_normalize_rewrites(raw, spoken):
    assert normalize_for_tts(raw) == spoken


@pytest.mark.parametrize(
    "prose",
    [
        "Alright boss, here's the full rundown on where things stand this morning.",
        "That's well-known, isn't it? The U.S. team won.",
        "Hey [laugh] that is funny [sigh] honestly.",
        "Hmm, shh, why try? My gym has rhythm.",
    ],
)
def test_plain_prose_is_untouched(prose):
    assert normalize_for_tts(prose) == prose


def test_non_english_is_untouched():
    assert normalize_for_tts("Il a 37% de chance.", language="fr") == "Il a 37% de chance."
