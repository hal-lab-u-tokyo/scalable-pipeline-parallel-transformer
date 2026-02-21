"""Experiment registry and re‑exports."""
from __future__ import annotations

from typing import Dict, Type

from .base import Experiment  # noqa: E402  (local import after forward‑ref)

EXP_REGISTRY: Dict[str, Type[Experiment]] = {}


def register(name: str):
    """Decorator to register an Experiment subclass."""

    def _wrap(cls: Type[Experiment]):
        if name in EXP_REGISTRY:
            raise KeyError(f"Experiment '{name}' is already registered")
        EXP_REGISTRY[name] = cls
        return cls

    return _wrap


# The concrete experiment modules import *this* module, so we place the
# imports at the end to avoid circular dependencies.
from .training.main import TrainingExperiment  # noqa: E402,F401
#from .actv_err.main import ActvErrExperiment  # noqa: E402,F401
from .profiling.main import ProfilingExperiment  # noqa: E402,F401]
from .actv_err.main import ActvErrExperiment