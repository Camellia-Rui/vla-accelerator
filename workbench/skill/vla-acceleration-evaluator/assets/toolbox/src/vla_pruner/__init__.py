"""VLA-Pruner public API; model dependencies are loaded on demand."""

from .accelerator import AcceleratorConfig, VLAPrunerAccelerator

__all__ = ["AcceleratorConfig", "VLAPrunerAccelerator"]
