"""Stochastic figure-ground stimuli, built to the controls in README.md."""

from .config import SFGConfig
from .build import build, render
from .verify import verify

__all__ = ["SFGConfig", "build", "render", "verify"]
