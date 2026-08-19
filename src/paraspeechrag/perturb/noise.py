"""Noise-axis perturbations: white Gaussian, ambient (ESC-50), reverberation.

Conventions that a replicator needs to know
-------------------------------------------

**SNR convention.** ``add_white_noise`` and ``add_ambient_noise`` define SNR
over the **full-signal RMS** of the utterance::

    P_signal = mean(x**2)                      # includes silence and pauses
    P_noise  = P_signal / 10**(snr_db / 10)

This is *not* the ITU-T P.56 active-speech level. Because Spoken-SQuAD
paragraph readings contain substantial silence, full-signal RMS understates
the speech level, so the effective speech-to-noise ratio is **higher** than the
nominal ``snr_db``. Any comparison against a paper that uses P.56 active level
will not line up. Do not change this convention without regenerating every
noise result.

**Clipping.** Both additive functions clip the sum to ``[-1, 1]``. At low SNR
this is itself a nonlinear distortion on top of the additive noise.

**Reverberation is not parameterised by SNR.** ``add_reverberation`` takes a
decay time, not an SNR. The evaluation driver
(``scripts/run_noise_robustness_eval.py``) reuses the ``--snr-levels`` values
as a decay-time knob via ``decay_time_ms = snr * 10``, so a row labelled
"reverb @ 20 dB" is really "reverb with a 200 ms decay tail". Reverb rows
therefore do **not** share an axis with the white/ambient rows. See
``docs/GAPS.md``.

**Determinism.** Every stochastic function takes an explicit
``rng: numpy.random.Generator``. Passing ``None`` creates a fresh unseeded
generator, which makes the output irreproducible — always thread a seeded
generator through from the caller.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.signal


def _as_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def scan_esc50_files(esc50_dir: str | Path) -> list[Path]:
    """Return the sorted list of WAV paths from an ESC-50 audio directory.

    Sorting is what makes a seeded draw over this list reproducible across
    machines, so do not switch to an unordered ``glob``.
    """
    esc50_dir = Path(esc50_dir)
    candidates = [esc50_dir / "audio", esc50_dir]
    for d in candidates:
        files = sorted(d.glob("*.wav"))
        if files:
            return files
    raise FileNotFoundError(
        f"No WAV files found in {esc50_dir} or {esc50_dir / 'audio'}. "
        "Run scripts/download_esc50.sh first."
    )


def load_esc50_clip(
    esc50_files: list[Path],
    target_sr: int = 16000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Load one ESC-50 clip and resample to ``target_sr`` (ESC-50 is 44.1 kHz)."""
    import soundfile as sf

    rng = _as_rng(rng)
    wav_path = esc50_files[int(rng.integers(len(esc50_files)))]
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        n_samples = int(len(audio) * target_sr / sr)
        audio = scipy.signal.resample(audio, n_samples)
    return audio.astype(np.float32)


def add_white_noise(
    audio: np.ndarray,
    snr_db: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add white Gaussian noise at ``snr_db`` (full-signal RMS convention).

    Args:
        audio: Input audio as a float32 array.
        snr_db: Signal-to-noise ratio in dB. Higher = less noise.
        rng: Seeded generator. ``None`` yields irreproducible output.

    Returns:
        Noisy audio, clipped to ``[-1, 1]``.
    """
    audio = np.asarray(audio, dtype=np.float32)
    rng = _as_rng(rng)

    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio

    noise_power = signal_power / (10 ** (snr_db / 10.0))

    noise = rng.standard_normal(len(audio)).astype(np.float32)
    noise_rms = np.sqrt(np.mean(noise ** 2))
    if noise_rms > 0:
        noise = noise * np.sqrt(noise_power) / noise_rms

    noisy_audio = audio + noise
    return np.clip(noisy_audio, -1.0, 1.0).astype(np.float32)


def add_ambient_noise(
    audio: np.ndarray,
    noise_audio: np.ndarray,
    snr_db: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add an ambient clip at ``snr_db`` (full-signal RMS convention).

    ``noise_audio`` must already be at the same sample rate as ``audio``. If it
    is shorter it is tiled; if longer, a window is cropped at a random offset
    drawn from ``rng`` — that offset is the second source of nondeterminism
    after the clip choice itself.
    """
    audio = np.asarray(audio, dtype=np.float32)
    noise_audio = np.asarray(noise_audio, dtype=np.float32)
    rng = _as_rng(rng)

    if len(noise_audio) < len(audio):
        n_repeats = (len(audio) // len(noise_audio)) + 1
        noise_audio = np.tile(noise_audio, n_repeats)[: len(audio)]
    else:
        start_idx = int(rng.integers(0, len(noise_audio) - len(audio) + 1))
        noise_audio = noise_audio[start_idx : start_idx + len(audio)]

    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio

    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise_rms = np.sqrt(np.mean(noise_audio ** 2))
    if noise_rms > 0:
        noise_audio = noise_audio * np.sqrt(noise_power) / noise_rms

    noisy_audio = audio + noise_audio
    return np.clip(noisy_audio, -1.0, 1.0).astype(np.float32)


def add_reverberation(
    audio: np.ndarray,
    decay_time_ms: float = 150.0,
    sr: int = 16000,
) -> np.ndarray:
    """Convolve with a synthetic exponential-decay impulse response.

    This is a hand-written IR, not a measured or simulated room: an
    exponentially decaying envelope over ``decay_time_ms``, with the direct
    sound forced to 1.0 and a single +0.5 early reflection at a fixed 50 ms.
    Note that the 50 ms reflection is placed at an absolute offset regardless
    of ``decay_time_ms``, so for ``decay_time_ms < 50`` it falls outside the IR
    and is silently dropped.

    The result is peak-normalised to 0.95 before clipping, which means this
    function also changes the signal's gain — unlike the additive noise
    functions.

    Deterministic: no randomness is involved.
    """
    audio = np.asarray(audio, dtype=np.float32)

    decay_samples = int(decay_time_ms / 1000.0 * sr)
    decay_samples = max(1, decay_samples)

    t = np.arange(decay_samples, dtype=np.float32) / sr
    rir = np.exp(-3.0 * t / (decay_time_ms / 1000.0))

    rir[0] = 1.0  # direct sound
    early_idx = max(1, int(0.05 * sr))  # 50 ms early reflection
    if early_idx < len(rir):
        rir[early_idx] += 0.5

    rir = rir / np.max(np.abs(rir))

    reverb_audio = scipy.signal.fftconvolve(audio, rir, mode="same")

    max_val = np.max(np.abs(reverb_audio))
    if max_val > 0:
        reverb_audio = reverb_audio / max_val * 0.95

    return np.clip(reverb_audio, -1.0, 1.0).astype(np.float32)
