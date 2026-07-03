"""
kmouflage/models/k_functions.py
================================
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
from scipy.optimize import brentq

from .. import equations as eq


def attractor_u_ini(solver) -> float:
    """
    Attractor solution for ũ_ini in the radiation-dominated era (eq. 300/305).
    Uses solver._phys (built by KMouflageBackground.run() before u_ini is
    called) and the pure equations.Z/F functions rather than private solver
    methods.
    """
    s     = solver
    phys  = s._phys
    phi0  = s.ic.phi_ini
    a_ini = np.exp(s.N_ini)

    alpha0     = phys.alpha(phi0)
    alpha_phi0 = phys.alpha_phi(phi0)
    F_ini      = eq.F(phys, phi0)

    # Conformal Hubble in RDE: H_conf ≈ √Ω_{r,0} · a⁻¹
    E_ini = np.sqrt(s.cosmo.Omega_r0) * a_ini**(-1)

    # Attractor constant R (eq. 300) — constant throughout the RDE
    R = -3.0 * alpha0 * s.cosmo.Omega_m0 / (2.0 * np.sqrt(s.cosmo.Omega_r0))

    if R == 0.0:
        return 0.0

    def g(phi_prime):
        Z_  = eq.Z(phys, phi0, phi_prime, a_ini)
        val = Z_ * phi_prime - R
        # field-dependent coupling correction (eq. 305)
        if alpha_phi0 != 0.0:
            val += 3.0 * F_ini * alpha0 * alpha_phi0 * phi_prime**2 / E_ini
        return val

    # Linear seed: φ'_lin = R / Z(X→0), gives the initial bracket endpoint
    Z0 = eq.Z(phys, phi0, 0.0, a_ini)
    if abs(Z0) < 1e-30:
        Z0 = 1.0
    phi_prime_lin = R / Z0

    # Bracket: [phi_prime_lin, 0] for R<0, [0, phi_prime_lin] for R>0.
    # g(0) = -R has opposite sign to g(phi_prime_lin) in the screened regime.
    lo = min(phi_prime_lin, 0.0)
    hi = max(phi_prime_lin, 0.0)

    # Safety expansion of the non-zero endpoint until sign change confirmed
    for _ in range(80):
        if g(lo) * g(hi) < 0:
            break
        if R < 0:
            lo *= 2.0
        else:
            hi *= 2.0
    else:
        raise RuntimeError(
            f"attractor_u_ini: bracket failed for model={s.model.name!r}, "
            f"phi0={phi0:.3e}, a_ini={a_ini:.3e}, R={R:.3e}"
        )

    phi_prime_ini = brentq(g, lo, hi, xtol=1e-12, rtol=1e-10)
    return float(phi_prime_ini / E_ini)


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


# ---------------------------------------------------------------------------
# Power-law model   K(X) = -1 + X + K0 * X^m
# ---------------------------------------------------------------------------

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

    # ---- K, K', K'' in terms of the Einstein-frame variable X ----
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
