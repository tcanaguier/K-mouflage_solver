"""
K-mouflage kinetic functions K(X), K'(X), K''(X) and their initial conditions.

Each model is packaged as a ``KModel`` dataclass whose ``u_ini`` callable
returns the dimensionless initial field velocity ũ = dφ̃/dN at z_ini.

Convention (Section 4, report):
    φ̃  = φ / M_Pl,0
    ũ  = dφ̃/dN = φ̄' / (M_Pl,0 · H_0 · E)
    X  = A²(φ) · χ_J    (Einstein-frame kinetic variable)
    χ_J = φ̄'² / (2 M̃⁴ · a² · H_0² · E²)   in dimensionless units
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from .. import equations as eq


def attractor_u_ini(solver) -> float:
    """
    Solver-facing adapter matching KModel.u_ini's u_ini(solver)->float
    contract. The actual physics — the radiation-dominated attractor solve
    (eq. 300/305) — is stateless and lives in equations.attractor_u_ini,
    which only needs (phys, phi0, a_ini, Omega_m0, Omega_r0) and is
    reusable/testable independently of the solver.
    """
    s = solver
    return eq.attractor_u_ini(
        s._phys, s.ic.phi_ini, np.exp(s.N_ini), s.cosmo.Omega_m0, s.cosmo.Omega_r0,
    )


@dataclass
class KModel:
    """
    Container for a K-mouflage kinetic model.

    Attributes
    ----------
    name : str
    K    : Callable[[float], float]   K(X)
    Kp   : Callable[[float], float]   dK/dX  (K'(X))
    Kpp  : Callable[[float], float]   d²K/dX²
    u_ini: Callable[[object], float]
        Maps a ``KMouflageBackground`` instance to the dimensionless initial
        field velocity ũ_ini = (dφ̃/dN)_ini.
    params : dict
        Free parameters for display / comparison.
    """
    name:   str
    K:      Callable[[float], float]
    Kp:     Callable[[float], float]
    Kpp:    Callable[[float], float]
    u_ini:  Callable          # u_ini(solver) -> float
    params: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"KModel({self.name})"


def make_powerlaw_K(K0: float, m: int) -> KModel:
    """
    Build a power-law K-model:

        K(X)   = -1 + X + K0 · X^m
        K'(X)  = 1  + m · K0 · X^(m-1)
        K''(X) = m(m-1) · K0 · X^(m-2)   [0 if m < 2]

    The initial velocity ũ_ini is derived from the attractor solution in
    the radiation/matter dominated era.

    Parameters
    ----------
    K0 : float   Coefficient of the X^m term.
    m  : int     Exponent (integer ≥ 1; m=3 recommended).
    """
    if m < 1:
        raise ValueError(f"Exponent m must be ≥ 1, got {m}.")

    def K(X: float) -> float:
        return -1.0 + X + K0 * X**m

    def Kp(X):
        Xc = np.maximum(X, 1e-300)   # max → np.maximum
        return 1.0 + m * K0 * Xc ** (m - 1)

    def Kpp(X):
        if m < 2:
            return 0.0
        Xc = np.maximum(X, 1e-300)   # max → np.maximum
        return m * (m - 1) * K0 * Xc ** (m - 2)

    return KModel(
        name   = f"power-law (K0={K0}, m={m})",
        K      = K,
        Kp     = Kp,
        Kpp    = Kpp,
        u_ini  = attractor_u_ini,
        params = {"K0": K0, "m": m},
    )



def make_LambdaCDM_K() -> KModel:
    """K constant (K=-1) → no kinetic term, φ̃ frozen (ũ=0)."""
    def K(X: float)   -> float: return -1.0
    def Kp(X: float)  -> float: return  0.0
    def Kpp(X: float) -> float: return  0.0

    def u_ini(solver) -> float:
        """φ̃ is frozen → ũ = 0 exactly."""
        return 0.0

    return KModel(
        name   = "ΛCDM (K= -1 )",
        K      = K,
        Kp     = Kp,
        Kpp    = Kpp,
        u_ini  = u_ini,
        params = {"None": None},
    )


def make_arctan_K(K_star, X_star):

    def K(X):   return -1.0 + X + K_star * (X - X_star * np.arctan(X / X_star))
    def Kp(X):  return 1.0 + K_star * (1.0 - 1.0 / (1.0 + (X / X_star)**2))
    def Kpp(X): return K_star * (2.0 * X / X_star**2) / (1.0 + (X / X_star)**2)**2

    return KModel(
        name   = f'arctan (K_star={K_star}, X_star={X_star})',
        K      = K,
        Kp     = Kp,
        Kpp    = Kpp,
        u_ini  = attractor_u_ini,
        params = {'K_star': K_star, 'X_star': X_star},
    )
