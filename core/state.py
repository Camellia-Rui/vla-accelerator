"""Episode lifecycle. Every pruner gets reset() called at episode start."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, List


class EpisodeState:
    """Tracks per-episode counters and gives every pruner a reset boundary."""

    def __init__(self) -> None:
        self.episode_id: int = 0
        self.step_id: int = 0
        self.is_first_frame: bool = True

    def advance(self) -> None:
        self.step_id += 1
        self.is_first_frame = False

    def start_new_episode(self) -> None:
        self.episode_id += 1
        self.step_id = 0
        self.is_first_frame = True


class StatefulPrunerMixin:
    """Mixin for pruners that carry cross-frame state (SpecPrune, VLA-Pruner, VLA-Cache)."""

    def reset(self) -> None:
        raise NotImplementedError


@contextmanager
def episode_boundary(pruner: Any, state: EpisodeState) -> Iterator[EpisodeState]:
    """Wrap one episode. Always calls pruner.reset() on entry."""
    state.start_new_episode()
    if hasattr(pruner, "reset"):
        pruner.reset()
    try:
        yield state
    finally:
        if hasattr(pruner, "reset"):
            pruner.reset()