"""Public API of the reviewer package.

Top-level names (`review_patch`, `Finding`, `build_client`) are resolved
lazily so that importing sibling modules like `reviewer.utils` does not pull
in the azure-ai-inference SDK. This keeps the unit tests — which only touch
pure helpers — runnable without the model SDK installed.
"""
from __future__ import annotations

__all__ = ["review_patch", "Finding", "build_client"]


def __getattr__(name: str):
    if name in ("review_patch", "Finding"):
        from .reviewer import review_patch, Finding
        return {"review_patch": review_patch, "Finding": Finding}[name]
    if name == "build_client":
        from .client import build_client
        return build_client
    raise AttributeError(f"module 'reviewer' has no attribute {name!r}")
