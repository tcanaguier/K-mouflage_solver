"""
Post-run diagnostics for a KMouflageBackground.

Work in progress: the full diagnostic report (observational bounds,
stability conditions, numerical consistency checks) is still being
developed and is not included in this snapshot yet.
"""

from __future__ import annotations

from .solver import KMouflageBackground


def verify(bg: KMouflageBackground) -> bool:
    """Placeholder — real checks are still in development."""
    if not hasattr(bg, 'phi'):
        raise RuntimeError("Call bg.run() before verify(bg).")
    print("[verify] work in progress, no checks implemented yet.")
    return True
