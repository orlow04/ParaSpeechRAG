"""Perturbation generators.

Only the **noise** axis is implemented here. The speaker, emotion, codec and
rate axes reported in the paper are *not* in this package — see
``docs/GAPS.md`` for what exists, what was produced out-of-tree, and what has
no implementation at all.
"""

from .noise import (
    add_ambient_noise,
    add_reverberation,
    add_white_noise,
    load_esc50_clip,
    scan_esc50_files,
)

__all__ = [
    "add_ambient_noise",
    "add_reverberation",
    "add_white_noise",
    "load_esc50_clip",
    "scan_esc50_files",
]
