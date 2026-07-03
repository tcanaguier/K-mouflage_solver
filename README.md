# Kmouflage — K-mouflage background & growth solver

Solveur numérique pour des modèles cosmologiques K-mouflage (théorie
scalaire-tenseur avec un terme cinétique non standard K(X) et un couplage
conforme au secteur matière), avec comparaison à ΛCDM. Le package résout
l'évolution de fond (champ scalaire φ, taux de Hubble, densités et
équations d'état) puis la croissance linéaire des structures (f, D+,
fσ8), et fournit des outils de calibration et de vérification physique.

## Structure du dépôt

```
kmouflage/                       # package Python (le solveur)
├── models/
│   ├── k_functions.py           # KModel : K(X), K'(X), K''(X) + conditions initiales
│   ├── couplings.py             # ConformalCoupling : A(φ), α(φ), α'(φ)
│   └── potential.py             # Potential : V(φ), V'(φ) — optionnel, désactivé par défaut
├── equations.py                 # Physique pure (F, X, ρ_φ, p_φ, Z, Z_eff, système linéaire...)
├── solver.py                    # CosmologicalParams, InitialConditions, KMouflageBackground
│                                 #   (intégration ODE + post-traitement + get_physical)
├── calibrate_M4.py               # calibrate_M4_tilde(bg, ...) : calibre M4_tilde sur Ω_DE(0)
├── verify.py                    # verify(bg) : rapport de diagnostics physiques/numériques
├── growth.py                    # GrowthSolver (base commune) + KmouflageGrowth + LCDMGrowth
└── io_utils.py                  # save_run() / load_run() : param.ini + data.npz

examples/
├── example_background.ipynb     # comparaison power-law / arctan / ΛCDM
├── comparison_potential.ipynb   # sans / avec potentiel de quintessence, fσ8 vs DESI DR1
└── results/                     # sorties de save_run() (param.ini + data.npz par run)
```

Le solveur numérique (`kmouflage/`) est entièrement séparé des notebooks
d'exemple (`examples/`).

## Dépendances

`numpy`, `scipy`, `matplotlib`.

## Usage minimal

```python
from kmouflage import (
    KMouflageBackground, CosmologicalParams,
    make_powerlaw_K, make_exponential_coupling, make_exponential_potential,
    calibrate_M4_tilde, verify,
    KmouflageGrowth, LCDMGrowth,
    save_run, load_run,
)

# ── modèle et couplage ────────────────────────────────────────────────────
model    = make_powerlaw_K(K0=1, m=3)
coupling = make_exponential_coupling(beta=0.1)
cosmo    = CosmologicalParams(H0_input=67.36, Omega_m0=0.25)

# ── fond, sans potentiel (comportement par défaut) ────────────────────────
bg = KMouflageBackground(model=model, coupling=coupling, cosmo=cosmo)
bg.run()
verify(bg)

# ── fond, avec un potentiel de quintessence (optionnel) ───────────────────
bg_quint = KMouflageBackground(
    model=model, coupling=coupling, cosmo=cosmo,
    potential=make_exponential_potential(V0=0.7, lam=1.0),
)
calibrate_M4_tilde(bg_quint, target_Omega_DE=0.75)

# ── croissance linéaire ────────────────────────────────────────────────────
kg = KmouflageGrowth(bg, N_points=2000)
results_k = kg.run()

lcdm = LCDMGrowth(Omega_m0=cosmo.Omega_m0, Omega_r0=cosmo.Omega_r0)
results_l = lcdm.run()

# ── sauvegarde / rechargement d'un run ─────────────────────────────────────
path   = save_run(bg, "examples/results", "power_law_beta0.1")
loaded = load_run(path)   # {"params": ConfigParser, "data": dict[str, ndarray]}
```

Le potentiel scalaire (`potential=...`) est optionnel : par défaut,
`KMouflageBackground` utilise un potentiel nul (`V=0`), au même titre que
`model` et `coupling` sont des objets injectables (voir `models/`).

## Notebooks

- `examples/example_background.ipynb` : comparaison des modèles power-law,
  arctan et ΛCDM (taux de Hubble, Ω_i(z), équations d'état, φ(z), μ_K(z)).
- `examples/comparison_potential.ipynb` : impact d'un potentiel de
  quintessence (calibration de M4_tilde par λ, écarts au modèle nu,
  fσ8(z) comparé à DESI DR1).

Les sorties sauvegardées via `save_run()` sont écrites dans
`examples/results/<nom_du_run>/` (un `param.ini` lisible + un `data.npz`).
