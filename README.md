# K-Mouflage Background

This is a small Python solver for K-mouflage models, a modified gravity
theory built on a scalar field with a non-standard kinetic term. It
computes how the universe expands (Hubble rate, densities, equations of
state) and how structures grow over time, and compares the result to
standard ΛCDM.

## Install

Just clone the repo and import `kmouflage` from it.

Needs: `numpy`, `scipy`, `matplotlib`.

## Quick example

```python
from kmouflage import (
    KMouflageBackground, CosmologicalParams,
    make_powerlaw_K, make_exponential_coupling,
)

model    = make_powerlaw_K(K0=1, m=3)
coupling = make_exponential_coupling(beta=0.1)
cosmo    = CosmologicalParams(omega_m=0.1430, omega_r=4.15e-5)   # Planck 2018 defaults

bg = KMouflageBackground(model=model, coupling=coupling, cosmo=cosmo)
bg.run()

N = 0.0          # N = ln(a), N=0 is today
bg.phi(N)        # scalar field
bg.Omega_m(N)    # matter density parameter
```

`H0`/`Omega_m0`/`Omega_de0` are *outputs* of the run (`bg.H0_predicted`,
`bg.Omega_m0_predicted`, `bg.Omega_de0_predicted`). To hit a target `H0`
instead, adjust `M4_tilde` with `calibrate_M4_tilde` (see
`kmouflage/calibrate_M4.py`).

`run_background(model, coupling, potential=None, cosmo=None, ic=None, verbose=False)`
does the same build-then-`run()` in one call:

```python
from kmouflage import run_background

bg = run_background(make_powerlaw_K(K0=1, m=3), make_exponential_coupling(beta=0.1))
```

## Saving a run to disk

`save_table(bg, name=None, fields=None, outdir=...)` writes a single CSV:
a commented header with the run's model/coupling/potential/cosmology, then
the chosen fields (default: a standard subset, see `kmouflage/run.py`'s
`DEFAULT_FIELDS`) as columns. `name` defaults to `default_table_name(bg)`
(`kmouflage_table_<coupling>_<model>_<potential>`, built from the model's
actual parameters) if not given. No loader is provided. Read the CSV with
whatever tool you like (`numpy.genfromtxt`, `pandas.read_csv`, ...).

```python
from kmouflage import run_background, save_table

bg = run_background(model, coupling, cosmo=cosmo)
save_table(bg)   # -> runs/kmouflage_table_....csv
```

## What's in the repo

- `kmouflage/` is the solver itself (the actual package).
- `examples/` has the Jupyter notebooks showing how to use it.
- `runs/` holds the saved results, created when you use `save_table`.

## Notebooks

- `example_run_background.ipynb` covers the essentials: build a model, run
  it, verify it, calibrate it, compare it, save it.
- `custom_models_guide.ipynb` shows how to build your own K(X) model or coupling.


## Papers

- (to add)

## Status

Still a work in progress:

- `verify.py`, the sanity-check module, isn't finished yet 
- No automated tests or packaging yet.
- Verbosity is just on/off for now; proper log levels are planned.
