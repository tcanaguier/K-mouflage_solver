from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import interp1d

from . import equations as eq
from .equations import Physics
from .models.k_functions import KModel
from .models.couplings import ConformalCoupling
from .models.potential import Potential, make_constant_potential


@dataclass
class CosmologicalParams:
    H100:       float = 100.0     # fixed reference unit [km/s/Mpc]

    Mpl:        float = 1.0
    M_Pl_input: float = 2.435e18  # unit [GeV]

    omega_m:    float = 0.1430    # Omega_m0 * h^2 (Planck 2018), fixed physical density
    omega_r:    float = 4.15e-5   # Omega_r0 * h^2 (Planck 2018), fixed physical density

    M4_tilde:   float = 1.0       # amplitude of the K-mouflage kinetic term


@dataclass
class InitialConditions:
    z_ini:   float = 1e5
    phi_ini: float = 0.0


class KMouflageBackground:
    def __init__(
        self,
        model:         KModel,
        coupling:      ConformalCoupling,
        cosmo:         CosmologicalParams | None = None,
        ic:            InitialConditions  | None = None,
        potential:     Potential          | None = None,
        rtol:          float = 1e-10,
        atol:          float = 1e-12,
        max_step:      float = 0.005,
        N_points:      int   = 100_000,
    ) -> None:
        self.model     = model
        self.coupling  = coupling
        self.cosmo     = cosmo or CosmologicalParams()
        self.ic        = ic    or InitialConditions()
        self.potential = potential or make_constant_potential()
        self.rtol      = rtol
        self.atol      = atol
        self.max_step  = max_step
        self.N_points  = N_points

        c = self.cosmo
        self.M4_tilde     = c.M4_tilde
        self.rho_m0_tilde = 3.0 * c.omega_m
        self.rho_r0_tilde = 3.0 * c.omega_r

        self.aeq   = c.omega_r / c.omega_m
        self.teq   = self.aeq**2 / (2.0 * np.sqrt(c.omega_r))
        self.N_ini = np.log(1.0 / (1.0 + self.ic.z_ini))
        self.N_end = 0.0

        self._phys: Physics | None = None

    def _build_physics(self) -> Physics:
        """Rebuilt on every run() so a mutation of M4_tilde/model/coupling/potential (e.g. by calibrate_M4_tilde) is picked up."""
        return Physics(
            model        = self.model,
            coupling     = self.coupling,
            potential    = self.potential,
            M4_tilde     = self.M4_tilde,
            rho_m0_tilde = self.rho_m0_tilde,
            rho_r0_tilde = self.rho_r0_tilde,
        )

    def _phi_prime_fixed_point(self, phi, u, a):
        """Fixed-point iteration phi_prime = E_conf(phi, phi_prime, a) * u, since E_conf depends on phi_prime through rho_phi."""
        phi_prime = 0.0
        E_conf = None
        for _ in range(20):
            E_conf        = eq.E_conf_from_F1(self._phys, phi, phi_prime, a)
            phi_prime_new = E_conf * u
            if abs(phi_prime_new - phi_prime) / (abs(phi_prime) + 1e-300) < 1e-12:
                break
            phi_prime = phi_prime_new
        return phi_prime, E_conf

    def _rhs_N(self, N, Y):
        phi, u = Y
        a      = np.exp(N)

        phi_prime, E_conf = self._phi_prime_fixed_point(phi, u, a)
        H_conf = E_conf

        M, phi_prime2, H_conf_prime = eq.linear_system(self._phys, phi, phi_prime, a, H_conf)

        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if not np.isfinite(det) or abs(det) < 1e-30:
            raise RuntimeError(f"Système singulier N={N:.4f} det={det:.3e}")

        dphi_dN = u
        du_dN   = (phi_prime2 / E_conf**2
                   - u * H_conf_prime / E_conf**2)

        return np.array([dphi_dN, du_dN])

    def _integrate_once(self, rtol=None, atol=None, N_points=None):
        rtol     = rtol     or self.rtol
        atol     = atol     or self.atol
        N_points = N_points or self.N_points

        phi_ini = self.ic.phi_ini
        u_ini   = self.model.u_ini(self)

        sol = solve_ivp(
            self._rhs_N,
            t_span   = (self.N_ini, self.N_end),
            y0       = np.array([phi_ini, u_ini]),
            method   = "Radau",
            t_eval   = np.linspace(self.N_ini, self.N_end, N_points),
            rtol     = rtol,
            atol     = atol,
            max_step = self.max_step,
        )
        if not sol.success:
            raise RuntimeError(f"Intégration échouée: {sol.message}")
        return sol

    def run(self, verbose=True) -> None:
        self._phys = self._build_physics()

        if verbose:
            print(f"\n[RUN] z_ini={self.ic.z_ini:.1e} | "
                  f"omega_m={self.cosmo.omega_m} | omega_r={self.cosmo.omega_r}")

        sol = self._integrate_once()
        self._build_interpolators(sol)

        self.h_predicted         = float(self.E_conf(0.0))
        self.H0_predicted        = self.h_predicted * self.cosmo.H100
        self.Omega_m0_predicted  = float(self.Omega_m(0.0))
        self.Omega_r0_predicted  = float(self.Omega_r(0.0))
        self.Omega_de0_predicted = float(self.Omega_de_def(0.0))

        if verbose:
            phi_f = sol.y[0, -1]
            M_Pl0_eff = np.sqrt(self._F(phi_f))
            print(f"[OK] φ̃(z=0)={phi_f:.4e} | M_Pl0_eff={M_Pl0_eff:.6f} | "
                  f"delta_Mpl={M_Pl0_eff - 1.0:+.4e}")
            print(f"[PREDICTED] h={self.h_predicted:.6f} | "
                  f"H0={self.H0_predicted:.4f} km/s/Mpc | "
                  f"Omega_m0={self.Omega_m0_predicted:.4f} | "
                  f"Omega_de0={self.Omega_de0_predicted:.4f}")

            Ode_ini = float(self.Omega_de_def(self._N[0]))
            print(f"[CHECK] Omega_DE(z_ini={self.ic.z_ini:.1e}) = {Ode_ini:.3e} "
                  f"(should be << 1 for the deep-RD omega_m/omega_r extrapolation to be valid)")
            if abs(Ode_ini) > 1e-3:
                print("[WARNING] Omega_DE not negligible at z_ini, possible tracking "
                      "behaviour; the RD-extrapolated initial conditions may not be safe "
                      "for this (K, f, lambda) combination.")

    def _F(self, phi): return eq.F(self._phys, phi)

    def _build_interpolators(self, sol) -> None:
        phys = self._phys

        N_arr   = sol.t
        phi_arr = sol.y[0]
        u_arr   = sol.y[1]
        a_arr   = np.exp(N_arr)

        def _interp(arr):
            return interp1d(N_arr, arr, kind='linear',
                            bounds_error=False, fill_value='extrapolate')

        E_conf_arr    = np.empty_like(N_arr)
        phi_prime_arr = np.empty_like(N_arr)

        for i, (ph, u, a) in enumerate(zip(phi_arr, u_arr, a_arr)):
            phi_prime_arr[i], E_conf_arr[i] = self._phi_prime_fixed_point(ph, u, a)

        H_conf_arr = E_conf_arr

        eta_arr      = cumulative_trapezoid(1.0 / H_conf_arr, N_arr, initial=0.0)
        t_cosmic_arr = cumulative_trapezoid(a_arr / H_conf_arr, N_arr, initial=0.0)
        # super-conformal time t_sc: dt = a^2 dt_sc  =>  dt_sc/dN = 1/(a*H_conf)
        t_superconform_arr = cumulative_trapezoid(1.0 / (a_arr * H_conf_arr), N_arr, initial=0.0)

        H_arr = eq.H(E_conf_arr, a_arr)

        A_arr  = np.asarray([phys.A(ph) for ph in phi_arr], dtype=float)
        av_arr = np.array([phys.alpha(ph)     for ph in phi_arr])
        ap_arr = np.array([phys.alpha_phi(ph) for ph in phi_arr])

        F_arr       = A_arr**(-2)
        F_phi_arr   = -2.0 * av_arr * F_arr
        F_phi2_arr  = 2.0 * F_arr * (2.0 * av_arr**2 - ap_arr)
        F_prime_arr = F_phi_arr * phi_prime_arr

        chi_arr   = phi_prime_arr**2 / (2.0 * self.M4_tilde * a_arr**2)
        X_arr     = A_arr**2 * chi_arr
        Kp_arr    = phys.Kp(X_arr)
        Kpp_arr   = phys.Kpp(X_arr)

        Z_arr     = A_arr**(-2) * Kp_arr - 6.0 * F_arr * av_arr**2
        Z_eff_arr = A_arr**(-2) * (Kp_arr + 2.0 * X_arr * Kpp_arr) - 6.0 * F_arr * av_arr**2

        phi_prime2_arr   = np.empty_like(N_arr)
        H_conf_prime_arr = np.empty_like(N_arr)
        for i, (ph, pp, a, Hc) in enumerate(zip(phi_arr, phi_prime_arr, a_arr, H_conf_arr)):
            _, phi_prime2_arr[i], H_conf_prime_arr[i] = eq.linear_system(phys, ph, pp, a, Hc)

        F_prime2_arr = F_phi2_arr * phi_prime_arr**2 + F_phi_arr * phi_prime2_arr

        A4_arr      = A_arr**(-4)
        f_arr       = np.asarray([phys.f_pot(ph) for ph in phi_arr], dtype=float)
        rho_phi_arr = ((Z_arr + 3.0 * F_arr * av_arr**2) * phi_prime_arr**2 / a_arr**2
                       - A4_arr * self.M4_tilde * (phys.K(X_arr) - f_arr))
        p_phi_arr   = (- 3.0 * F_arr * av_arr**2 * phi_prime_arr**2 / a_arr**2
                       + A4_arr * self.M4_tilde * (phys.K(X_arr) - f_arr))

        rho_m_arr = self.rho_m0_tilde * a_arr**(-3)
        rho_r_arr = self.rho_r0_tilde * a_arr**(-4)
        p_r_arr   = rho_r_arr / 3.0

        eps_F_arr    = 1 / F_arr - 1.0

        rho_de_fried_arr = (3.0  * H_conf_arr**2 / a_arr**2
                            - rho_m_arr - rho_r_arr)

        rho_de_def_arr = (eps_F_arr * (rho_m_arr + rho_r_arr)
                          + ( 1 / F_arr)
                          * (rho_phi_arr - 3.0 * H_conf_arr * F_prime_arr / a_arr**2))

        p_de_arr = (eps_F_arr * p_r_arr
                    + (1 / F_arr)
                    * (p_phi_arr + (F_prime2_arr + H_conf_arr * F_prime_arr) / a_arr**2))

        w_phi_arr      = np.where(np.abs(rho_phi_arr)      > 1e-30,
                                  p_phi_arr / rho_phi_arr,      -1.0)
        w_de_fried_arr = np.where(np.abs(rho_de_fried_arr) > 1e-30,
                                  p_de_arr / rho_de_fried_arr,  -1.0)
        w_de_def_arr   = np.where(np.abs(rho_de_def_arr)   > 1e-30,
                                  p_de_arr / rho_de_def_arr,    -1.0)

        denom         = 3.0 * H_conf_arr**2
        Om_arr        = a_arr**2 * rho_m_arr        / denom
        Or_arr        = a_arr**2 * rho_r_arr        / denom
        Op_arr        = a_arr**2 * rho_phi_arr      / denom
        Ode_fried_arr = a_arr**2 * rho_de_fried_arr / denom
        Ode_def_arr   = a_arr**2 * rho_de_def_arr   / denom

        # mu_K_arr_old = (1 / F_arr) * (1.0 + 2.0 * F_arr * av_arr**2 / Z_arr)

        mu_K_arr = ( 1 / F_arr) * (1.0 + (2.0 * F_arr * av_arr**2 )/ (Z_arr + 6 * F_arr * av_arr**2))

        res_arr      = (3.0 * (H_conf_arr**2 * F_arr + H_conf_arr * F_prime_arr) / a_arr**2
                        - rho_m_arr - rho_r_arr - rho_phi_arr)
        rho_tot_arr  = rho_m_arr + rho_r_arr + np.abs(rho_phi_arr)
        residual_arr = np.abs(res_arr) / (rho_tot_arr + 1e-30)

        det_arr = 2.0 * F_arr * Z_eff_arr - 6.0 * av_arr * F_arr * F_prime_arr
        M_Pl0_eff = F_arr[-1]
        self.M_Pl0_eff   = M_Pl0_eff
        self.delta_Mpl   = M_Pl0_eff - 1.0
        self.delta_H     = E_conf_arr[-1] - 1.0

        self.phi             = _interp(phi_arr)
        self.u               = _interp(u_arr)
        self.E_conf          = _interp(E_conf_arr)
        self.a               = _interp(a_arr)
        self.z               = _interp(1.0 / a_arr - 1.0)
        self.eta             = _interp(eta_arr)
        self.t_cosmic        = _interp(t_cosmic_arr)
        self.t_superconform  = _interp(t_superconform_arr)
        self.H               = _interp(H_arr)
        self.H_conf          = _interp(H_conf_arr)
        self.phi_prime       = _interp(phi_prime_arr)
        self.phi_prime2      = _interp(phi_prime2_arr)
        self.H_conf_prime    = _interp(H_conf_prime_arr)
        self.F               = _interp(F_arr)
        self.F_phi           = _interp(F_phi_arr)
        self.F_phi2          = _interp(F_phi2_arr)
        self.F_prime         = _interp(F_prime_arr)
        self.F_prime2        = _interp(F_prime2_arr)
        self.chi             = _interp(chi_arr)
        self.X               = _interp(X_arr)
        self.Z               = _interp(Z_arr)
        self.Z_eff           = _interp(Z_eff_arr)
        self.alpha           = _interp(av_arr)
        self.rho_m           = _interp(rho_m_arr)
        self.rho_r           = _interp(rho_r_arr)
        self.rho_phi         = _interp(rho_phi_arr)
        self.p_phi           = _interp(p_phi_arr)
        self.w_phi           = _interp(w_phi_arr)
        self.epsilon_F       = _interp(eps_F_arr)
        self.rho_de_def      = _interp(rho_de_def_arr)
        self.rho_de_fried    = _interp(rho_de_fried_arr)
        self.p_de            = _interp(p_de_arr)
        self.w_de_fried      = _interp(w_de_fried_arr)
        self.w_de_def        = _interp(w_de_def_arr)
        self.Omega_m         = _interp(Om_arr)
        self.Omega_r         = _interp(Or_arr)
        self.Omega_phi       = _interp(Op_arr)
        self.Omega_de_fried  = _interp(Ode_fried_arr)
        self.Omega_de_def    = _interp(Ode_def_arr)
        self.mu_K            = _interp(mu_K_arr)
        self.residual_F1     = _interp(residual_arr)
        self._N               = N_arr
        self._det             = det_arr

    def get_physical(self, z, M_Pl=None):
        _C_KM_S   = 299792.458          # speed of light [km/s]
        _GYR_UNIT = 977.8                # 1/H100 [Gyr]
        _MPC_UNIT = _C_KM_S              # c/H100 [Mpc]

        z  = np.asarray(z, dtype=float).reshape(-1)
        N  = np.log(1.0 / (1.0 + z))
        N0 = 0.0

        H100   = self.cosmo.H100                            # [km/s/Mpc], fixed reference unit
        if M_Pl is None:
            M_pl = self.cosmo.M_Pl_input                    # Planck mass [GeV] (usually)
        else:
            M_pl = M_Pl
        t_unit = _GYR_UNIT / H100                           # [Gyr] per reduced time unit
        d_unit = _C_KM_S   / H100                           # [Mpc] per reduced distance unit

        # Hubble: H_conf = aH  ->  H = H_conf / a = E_conf * (1+z)
        H_conf_phys = self.E_conf(N) * H100
        H_phys = self.E_conf(N) * (1.0 + z) * H100

        # Cosmic time: cumulative from z_ini, t_today fixed at N=0
        t_today    = float(self.t_cosmic(N0)) * t_unit     # [Gyr]
        t_lookback = (float(self.t_cosmic(N0)) - self.t_cosmic(N)) * t_unit

        # Comoving distance from us (z=0): eta increases from z_ini toward z=0
        eta0 = float(self.eta(N0))
        d_C  = (eta0 - self.eta(N)) * d_unit
        d_A  = d_C / (1.0 + z)
        d_L  = d_C * (1.0 + z)

        # Scalar field & Planck mass
        phi_phys  = self.phi(N) * M_pl                 # phi in GeV
        M_pl_eff = np.sqrt(self.F(N)) * M_pl           # effective cosmological M_pl [GeV]

        return {
            "z"          : z,
            "a"          : 1.0 / (1.0 + z),
            "H_phys"     : H_phys,
            "H_conf_phys": H_conf_phys,
            "t_today"    : t_today,
            "t_lookback" : t_lookback,
            "d_C"        : d_C,
            "d_A"        : d_A,
            "d_L"        : d_L,
            "phi_phys"   : phi_phys,
            "M_pl_eff"   : M_pl_eff,
            "w_de"       : self.w_de_def(N),
            "mu_K"       : self.mu_K(N),
            "Omega_m"    : self.Omega_m(N),
            "Omega_de"   : self.Omega_de_def(N),
        }
