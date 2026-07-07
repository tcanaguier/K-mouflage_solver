"""
Calibration externe de M4_tilde pour le modèle K-mouflage.

Usage
-----
>>> from kmouflage.solver import KMouflageBackground, CosmologicalParams
>>> from kmouflage.calibrate_M4 import calibrate_M4_tilde
>>>
>>> bg = KMouflageBackground(model, coupling, cosmo=CosmologicalParams(M4_tilde=0.75))
>>> M4, Ode0, nit = calibrate_M4_tilde(bg, target_Omega_DE=0.75)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .solver import KMouflageBackground


def calibrate_M4_tilde(
    bg: KMouflageBackground,
    target_Omega_DE: float = 0.75,
    tol: float = 1e-5,
    max_iter: int = 50,
    verbose: bool = True,
    bracket: tuple[float, float] = (0.1, 3.0),
) -> tuple[float, float, int]:
    """
    Calibre bg.M4_tilde pour que Omega_DE(z=0) == target_Omega_DE, par
    recherche de racine (brentq) sur M4_tilde dans bracket.

    La cible est bg.Omega_de_def(0), la définition physique complète qui contient
    tous les termes K-mouflage (cinétique, K, epsilon_F*(rho_m+rho_r), F'/a^2).
    phi_ini=0 reste fixe ; seul M4_tilde est ajusté.

    (Une variante point-fixe multiplicatif a existé mais ne convergeait pas
    de façon fiable ; brentq est la seule méthode désormais.)

    Parameters
    ----------
    bg : KMouflageBackground
        Solveur de fond déjà configuré (model, coupling, cosmo, ic).
        M4_tilde est modifié en place sur cet objet.
    target_Omega_DE : float
        Valeur cible de Omega_DE(z=0) (défaut 0.75).
    tol : float
        Tolérance de convergence sur |Omega_DE(0) - target| (défaut 1e-5).
    max_iter : int
        Nombre maximal d'itérations brentq (défaut 50).
    verbose : bool
        Affiche le tableau d'itérations si True.
    bracket : tuple[float, float]
        Intervalle [a, b] encadrant la racine : Omega_DE(a) < target < Omega_DE(b).

    Returns
    -------
    M4_tilde_final : float
    Omega_DE_0_final : float
    n_iter : int
        Nombre d'évaluations effectuées par brentq.
    """
    def _run_and_read(m4: float) -> float:
        if m4 <= 0.0:
            raise ValueError(f"M4_tilde = {m4:.3e} <= 0")
        bg.M4_tilde = m4
        bg.run(verbose=False)
        return float(bg.Omega_de_def(0.0))

    if verbose:
        header = f"{'it':>4}  {'M4_tilde':>16}  {'Omega_DE(0)':>14}  {'err':>13}"
        sep = "─" * len(header)
        print(f"\n[calibrate_M4_tilde]  target={target_Omega_DE}  tol={tol:.1e}")
        print(sep)
        print(header)
        print(sep)

    it_count = [0]

    def residual(m4: float) -> float:
        Ode0 = _run_and_read(m4)
        err = Ode0 - target_Omega_DE
        if verbose:
            print(f"{it_count[0]:>4}  {m4:>16.10f}  {Ode0:>14.8f}  {err:>+13.4e}")
        it_count[0] += 1
        return err

    m4_sol = brentq(residual, bracket[0], bracket[1], xtol=tol * 1e-3, rtol=1e-12,
                    maxiter=max_iter, full_output=False)
    Ode0_final = _run_and_read(m4_sol)
    n_iter = it_count[0]
    if verbose:
        print(sep)
        print(f"[brentq converged]  it={n_iter}  M4_tilde={m4_sol:.10f}  "
              f"Omega_DE(0)={Ode0_final:.8f}  err={Ode0_final - target_Omega_DE:+.4e}\n")
    return bg.M4_tilde, Ode0_final, n_iter
