from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Potential:
    """
    Optional scalar-field potential V(phi), added to the K-mouflage
    equations of motion (see equations.py). Defaults to no potential
    (V=0 everywhere) via make_no_potential(), matching the KMouflageBackground
    default.
    """
    name:   str
    V:      Callable[[float], float]
    V_phi:  Callable[[float], float]
    params: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Potential({self.name})"


def make_no_potential() -> Potential:
    return Potential(
        name   = "none (V=0)",
        V      = lambda phi: 0.0,
        V_phi  = lambda phi: 0.0,
        params = {},
    )


def make_exponential_potential(V0: float = 0.7, lam: float = 1.0) -> Potential:
    def V(phi):     return V0 * np.exp(-lam * phi)
    def V_phi(phi): return -lam * V(phi)
    return Potential(
        name   = f"exponential (V0={V0}, lam={lam})",
        V      = V,
        V_phi  = V_phi,
        params = {"V0": V0, "lam": lam},
    )
