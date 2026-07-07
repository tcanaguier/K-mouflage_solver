# Kmouflage — K-mouflage background & growth solver

This is a small Python solver for K-mouflage models — a modified gravity
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
cosmo    = CosmologicalParams(H0_input=67.36, Omega_m0=0.25)

bg = KMouflageBackground(model=model, coupling=coupling, cosmo=cosmo)
bg.run()

N = 0.0          # N = ln(a), N=0 is today
bg.phi(N)        # scalar field
bg.Omega_m(N)    # matter density parameter
```


## Saving and reusing runs

Running the solver takes a bit of coding lignes, so `io_utils.py` gives you a
simple way to avoid doing it again

```python
from kmouflage import run_solver

result = run_solver(model, coupling, cosmo=cosmo)
result["data"]["Omega_m"]   # the saved arrays, as numpy arrays
```

`run_solver` runs the model and saves it, or just reloads it if you've
already run that exact setup before. There's also `save_run(bg, name)` and
`load_run(path)` if you want to save/load a run by hand, and `find_runs()`
to list what you've already saved.

## What's in the repo

- `kmouflage/` — the solver itself (the actual package).
- `examples/` — Jupyter notebooks showing how to use it.
- `runs/` — cached results, created automatically when you use `run_solver`.

## Notebooks

- `example_background.ipynb` — compares a few K(X) models to ΛCDM.
- `custom_models_guide.ipynb` — how to build your own K(X) model or coupling.
- `comparison_potential.ipynb` — what changes when you add a quintessence
  potential


## Status

Still a work in progress:

- `verify.py`, the sanity-check module, isn't finished yet 
- No automated tests or packaging yet.
- Verbosity is just on/off for now; proper log levels are planned.
