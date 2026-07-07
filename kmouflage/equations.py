"""
Pure K-mouflage equations of motion (background level).

Every function here is stateless: it takes a ``Physics`` context (model +
coupling + potential + cosmological constants) plus the current field state
(phi, phi_prime, a, H_conf) and returns a physical quantity.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

if TYPE_CHECKING:
    from .models.couplings import ConformalCoupling
    from .models.k_functions import KModel
    from .models.potential import Potential


@dataclass(frozen=True)
class Physics:
    """Bundles the model, coupling, potential and cosmological constants
    needed to evaluate the K-mouflage equations of motion."""
    model:        "KModel"
    coupling:     "ConformalCoupling"
    potential:    "Potential"
    M4_tilde:     float
    rho_m0_tilde: float
    rho_r0_tilde: float

    def A(self, phi):         return self.coupling.A(phi)
    def alpha(self, phi):     return self.coupling.alpha(phi)
    def alpha_phi(self, phi): return self.coupling.alpha_phi(phi)
    def K(self, X):           return self.model.K(X)
    def Kp(self, X):          return self.model.Kp(X)
    def Kpp(self, X):         return self.model.Kpp(X)
    def V(self, phi):         return self.potential.V(phi)
    def V_phi(self, phi):     return self.potential.V_phi(phi)


def H(E_conf, a) -> float:
    """Physical Hubble rate normalized by H0: H/H0 = E_conf/a (E_conf = ℋ/H0)."""
    return E_conf / a


def F(phys: Physics, phi) -> float:
    return phys.A(phi) ** (-2)


def F_phi(phys: Physics, phi) -> float:
    return -2.0 * phys.alpha(phi) * F(phys, phi)


def F_phi2(phys: Physics, phi) -> float:
    return 2.0 * F(phys, phi) * (2.0 * phys.alpha(phi) ** 2 - phys.alpha_phi(phi))


def chi(phys: Physics, phi_prime, a) -> float:
    return phi_prime ** 2 / (2.0 * phys.M4_tilde * a ** 2)


def X(phys: Physics, phi, phi_prime, a) -> float:
    return phys.A(phi) ** 2 * chi(phys, phi_prime, a)


def rho_m(phys: Physics, a) -> float:
    return phys.rho_m0_tilde * a ** (-3)


def rho_r(phys: Physics, a) -> float:
    return phys.rho_r0_tilde * a ** (-4)


def p_r(phys: Physics, a) -> float:
    return rho_r(phys, a) / 3.0


def Z(phys: Physics, phi, phi_prime, a) -> float:
    X_ = X(phys, phi, phi_prime, a)
    return phys.A(phi) ** (-2) * phys.Kp(X_) - 6.0 * F(phys, phi) * phys.alpha(phi) ** 2


def Z_eff(phys: Physics, phi, phi_prime, a) -> float:
    X_ = X(phys, phi, phi_prime, a)
    return (phys.A(phi) ** (-2) * (phys.Kp(X_) + 2.0 * X_ * phys.Kpp(X_))
            - 6.0 * F(phys, phi) * phys.alpha(phi) ** 2)


def Z_phi(phys: Physics, phi, phi_prime, a) -> float:
    X_ = X(phys, phi, phi_prime, a)
    av = phys.alpha(phi)
    ap = phys.alpha_phi(phi)
    return (phys.A(phi) ** (-2) * (-2.0 * av * phys.Kp(X_) + 2.0 * av * X_ * phys.Kpp(X_))
            + 12.0 * F(phys, phi) * av * (av ** 2 - ap))


def rho_phi(phys: Physics, phi, phi_prime, a) -> float:
    X_ = X(phys, phi, phi_prime, a)
    Z_ = Z(phys, phi, phi_prime, a)
    F_ = F(phys, phi)
    av = phys.alpha(phi)
    A4 = phys.A(phi) ** (-4)
    return ((Z_ + 3.0 * F_ * av ** 2) * phi_prime ** 2 / a ** 2
            - A4 * phys.M4_tilde * phys.K(X_)
            + A4 * phys.V(phi))


def p_phi(phys: Physics, phi, phi_prime, a) -> float:
    X_ = X(phys, phi, phi_prime, a)
    F_ = F(phys, phi)
    av = phys.alpha(phi)
    A4 = phys.A(phi) ** (-4)
    return (- 3.0 * F_ * av ** 2 * phi_prime ** 2 / a ** 2
            + A4 * phys.M4_tilde * phys.K(X_)
            - A4 * phys.V(phi))


def E_conf_from_F1(phys: Physics, phi, phi_prime, a) -> float:
    F_       = F(phys, phi)
    F_prime  = F_phi(phys, phi) * phi_prime
    rho_tot  = rho_m(phys, a) + rho_r(phys, a) + rho_phi(phys, phi, phi_prime, a)
    disc     = 9.0 * F_prime ** 2 + 12.0 * F_ * a ** 2 * rho_tot
    if disc < 0:
        raise ValueError(f"Discriminant négatif: disc={disc:.3e}")
    H_conf = (-3.0 * F_prime + np.sqrt(disc)) / (6.0 * F_)
    if H_conf <= 0:
        raise ValueError(f"H_conf={H_conf:.3e} <= 0")
    return H_conf


def linear_system(phys: Physics, phi, phi_prime, a, H_conf):
    F_    = F(phys, phi)
    av    = phys.alpha(phi)
    ap    = phys.alpha_phi(phi)
    X_    = X(phys, phi, phi_prime, a)
    Z_    = Z(phys, phi, phi_prime, a)
    Zphi_ = Z_phi(phys, phi, phi_prime, a)
    Kp    = phys.Kp(X_)
    Kpp   = phys.Kpp(X_)
    A2    = phys.A(phi) ** (-2)
    A4    = phys.A(phi) ** (-4)

    M = np.array([
        [Z_eff(phys, phi, phi_prime, a), 6.0 * av * F_],
        [-F_phi(phys, phi),              2.0 * F_     ],
    ])

    V_  = phys.V(phi)
    Vp_ = phys.V_phi(phi)

    d1 = (
        - 2.0 * H_conf * (Z_ - A2 * X_ * Kpp) * phi_prime

        - (Zphi_ + 2.0 * av * A2 * Kpp * X_) * phi_prime ** 2

        + 6.0 * F_ * av * (ap - av ** 2) * phi_prime ** 2

        + av * A2 * Kp * phi_prime ** 2

        - 4.0 * av * A4 * phys.M4_tilde * phys.K(X_) * a ** 2

        - 6.0 * av * F_ * H_conf ** 2

        + a ** 2 * A4 * (4.0 * av * V_ - Vp_)
    )

    d2 = (- 2.0 * F_ * (2.0 * av ** 2 - ap) * phi_prime ** 2
          + 2.0 * av * F_ * H_conf * phi_prime
          - F_ * H_conf ** 2
          - a ** 2 * (p_r(phys, a) + p_phi(phys, phi, phi_prime, a)))

    sol_norm     = np.linalg.solve(M, np.array([d1, d2]))
    phi_prime2   = sol_norm[0]
    H_conf_prime = sol_norm[1]

    return M, phi_prime2, H_conf_prime


def attractor_u_ini(phys: Physics, phi0, a_ini, Omega_m0, Omega_r0) -> float:
    """
    Attractor solution for the initial dimensionless field velocity ũ_ini
    in the radiation-dominated era (eq. 300/305). Generic for any K(X)
    model that behaves like standard k-essence/quintessence (K(X) → X-1)
    near X=0 used as the default u_ini for the power-law and arctan
    KModels (see models/k_functions.py, which wraps this in a thin
    solver-facing adapter matching KModel.u_ini's u_ini(solver)->float
    contract).
    """
    alpha0     = phys.alpha(phi0)
    alpha_phi0 = phys.alpha_phi(phi0)
    F_ini      = F(phys, phi0)

    # Conformal Hubble in RDE: H_conf ≈ √Ω_{r,0} · a⁻¹
    E_ini = np.sqrt(Omega_r0) * a_ini**(-1)

    # Attractor constant R (eq. 300) — constant throughout the RDE
    R = -3.0 * alpha0 * Omega_m0 / (2.0 * np.sqrt(Omega_r0))

    if R == 0.0:
        return 0.0

    def g(phi_prime):
        Z_  = Z(phys, phi0, phi_prime, a_ini)
        val = Z_ * phi_prime - R
        # field-dependent coupling correction (eq. 305)
        if alpha_phi0 != 0.0:
            val += 3.0 * F_ini * alpha0 * alpha_phi0 * phi_prime**2 / E_ini
        return val

    # Linear seed: φ'_lin = R / Z(X→0), gives the initial bracket endpoint
    Z0 = Z(phys, phi0, 0.0, a_ini)
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
            f"attractor_u_ini: bracket failed for model={phys.model.name!r}, "
            f"phi0={phi0:.3e}, a_ini={a_ini:.3e}, R={R:.3e}"
        )

    phi_prime_ini = brentq(g, lo, hi, xtol=1e-12, rtol=1e-10)
    return float(phi_prime_ini / E_ini)
