"""
kmouflage/verify.py
====================
Post-run diagnostics for a KMouflageBackground.
"""

from __future__ import annotations

import numpy as np

from .solver import KMouflageBackground


def verify(bg: KMouflageBackground) -> bool:
    """
    Run all physical and numerical checks after bg.run(), print a report,
    and return True iff every check passes.

    Checks
    ------
    Observational:
      1. |ε_F| < 0.1 everywhere  (BBN / G_eff constraint, eq. 18)
      2. |α(φ_0)·φ̃'_0| ≤ 0.2·H_0 at z=0  (coupling-velocity bound)
    Numerical consistency:
      3. φ̃' = ũ · E_conf  (definition of ũ)
      4. rho_phi + p_phi = IMPLEmenter.
    Stability (eqs. 50–51):
      4. Z_eff > 0  (no ghost)
      5. Z > 0      (no gradient instability)
      6. |det M| > 0  (linear system non-singular)
    """
    if not hasattr(bg, 'phi'):
        raise RuntimeError("Call bg.run() before verify(bg).")

    N_arr = bg._N

    def _s(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    # ── 1. |ε_F| < 0.1 ──────────────────────────────────────────────────
    eps_F_arr = bg.epsilon_F(N_arr)
    max_eps_F = float(np.max(np.abs(eps_F_arr)))
    ok_eps_F  = max_eps_F < 0.1

    # ── 2. |α·φ̃'| ≤ 0.2 at z=0 ─────────────────────────────────────────
    N0          = 0.0
    alpha_0     = float(bg.alpha(N0))
    phi_prime_0 = float(bg.phi_prime(N0))
    coupling_0  = abs(alpha_0 * phi_prime_0)
    ok_coupling = coupling_0 <= 0.2

    # ── 3. φ̃' = ũ · E_conf  (conformal-time definition of ũ) ─────────────
    u_arr         = bg.u(N_arr)
    E_conf_arr    = bg.E_conf(N_arr)
    phi_prime_arr = bg.phi_prime(N_arr)
    residual_pp   = np.abs(phi_prime_arr - u_arr * E_conf_arr)
    max_res_pp    = float(np.max(residual_pp / (np.abs(phi_prime_arr) + 1e-300)))
    ok_deriv      = max_res_pp < 1e-6

    # ── 3b. ũ = dφ̃/dN  (numerical gradient w.r.t. N = ln a) ────────────
    # Independent check: differentiate the φ̃ interpolator on the uniform
    # N grid and compare to the directly integrated variable ũ.
    # np.gradient gives dφ̃/dN (≠ φ̃' which is dφ̃/dη = ũ·E_conf).
    phi_arr_v    = bg.phi(N_arr)
    num_dphi_dN  = np.gradient(phi_arr_v, N_arr)
    residual_u   = np.abs(num_dphi_dN - u_arr)
    max_res_u    = float(np.max(residual_u / (np.abs(u_arr) + 1e-300)))
    ok_num_grad  = max_res_u < 1e-4

    # ── 4. No ghost: Z_eff > 0 ───────────────────────────────────────────
    Z_eff_arr   = bg.Z_eff(N_arr)
    min_Z_eff   = float(np.min(Z_eff_arr))
    ok_no_ghost = min_Z_eff > 0.0

    # ── 5. No gradient instability: Z > 0 ────────────────────────────────
    Z_arr      = bg.Z(N_arr)
    min_Z      = float(np.min(Z_arr))
    ok_no_grad = min_Z > 0.0

    # ── 6. det(M) ≠ 0 ────────────────────────────────────────────────────
    min_abs_det = float(np.min(np.abs(bg._det)))
    ok_det      = min_abs_det > 1e-30

    # ── Initial conditions ────────────────────────────────────────────────
    N_i           = bg.N_ini
    phi_ini       = float(bg.phi(N_i))
    phi_prime_ini = float(bg.phi_prime(N_i))
    H_ini         = float(bg.E_conf(N_i))
    Z_ini         = float(bg.Z(N_i))
    Ode_ini       = float(bg.Omega_de_def(N_i))
    rho_phi_ini   = float(bg.rho_phi(N_i))
    p_phi_ini     = float(bg.p_phi(N_i))

    # ── Report ───────────────────────────────────────────────────────────
    W = 62
    bar  = "─" * W
    dbar = "═" * W

    print(f"\n{dbar}")
    print(f"  VERIFICATION REPORT")
    print(f"  Model   : {bg.model.name}")
    print(f"  Coupling: {bg.coupling.name}")
    print(dbar)

    print(f"\n  Observational constraints")
    print(f"  {bar}")
    print(f"  {'|ε_F| < 0.1  (BBN/G_eff)':<38}  max = {max_eps_F:+.3e}  [{_s(ok_eps_F)}]")
    print(f"  {'|α·φ̃′| ≤ 0.2·H₀  at z=0':<38}  val = {coupling_0:+.3e}  [{_s(ok_coupling)}]")

    print(f"\n  Numerical consistency")
    print(f"  {bar}")
    print(f"  {'φ̃′ = ũ·E_conf  (conformal time)':<38}  max_rel = {max_res_pp:.3e}  [{_s(ok_deriv)}]")
    print(f"  {'ũ = dφ̃/dN  (numerical gradient)':<38}  max_rel = {max_res_u:.3e}  [{_s(ok_num_grad)}]")

    print(f"\n  Stability conditions")
    print(f"  {bar}")
    print(f"  {'No ghost   Z_eff > 0':<38}  min = {min_Z_eff:+.3e}  [{_s(ok_no_ghost)}]")
    print(f"  {'No gradient  Z > 0':<38}  min = {min_Z:+.3e}  [{_s(ok_no_grad)}]")
    print(f"  {'Non-singular  |det M| > 0':<38}  min = {min_abs_det:.3e}  [{_s(ok_det)}]")

    print(f"\n  Initial conditions  (z_ini = {bg.ic.z_ini:.1e})")
    print(f"  {bar}")
    rows = [
        ("φ̃_ini",        phi_ini),
        ("φ̃′_ini",       phi_prime_ini),
        ("H_conf,ini",   H_ini),
        ("Z_ini",         Z_ini),
        ("Ω_DE,ini",     Ode_ini),
        ("ρ_φ,ini",      rho_phi_ini),
        ("p_φ,ini",      p_phi_ini),
    ]
    for label, val in rows:
        print(f"  {label:<20}  {val:+.6e}")

    all_ok = all([ok_eps_F, ok_coupling, ok_deriv, ok_num_grad, ok_no_ghost, ok_no_grad, ok_det])
    print(f"\n{dbar}")
    print(f"  Overall : {'ALL CHECKS PASS' if all_ok else 'SOME CHECKS FAILED'}")
    print(f"{dbar}\n")

    return all_ok


#Faire main pour tester fonction verif avec un modèle ou 2 modèles données