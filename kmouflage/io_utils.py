"""
Save / reload a KMouflageBackground run as a human-readable param.ini
(cosmology, model, coupling, potential, numerics) plus a data.npz of the
derived quantities on the solver's N grid, for reproducible comparisons
without re-integrating each time. Saved by default under <project root>/runs/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import configparser

import numpy as np

from .solver import KMouflageBackground
from .calibrate_M4 import calibrate_M4_tilde

# a, phi and phi_prime (together with N, always saved separately in
# save_run) are the minimum sufficient state to reconstruct any other
# background quantity later via equations.py's pure functions — every
# derived quantity (rho_phi, p_phi, Z, Z_eff, F, mu_K, ...) is a function
# of (phys, phi, phi_prime, a), given phys rebuilt from param.ini's
# model/coupling/potential/cosmo (the caller re-creates the same objects;
# param.ini documents them but doesn't auto-load them). Even H_conf is
# recoverable, via eq.E_conf_from_F1(phys, phi, phi_prime, a).
# save_run() always saves these regardless of `fields` — they are not
# optional, a custom `fields=` list cannot drop them.
MANDATORY_FIELDS = ["a", "phi", "phi_prime"]

DEFAULT_FIELDS = [
    "z", "a", "phi", "phi_prime",
    "t_cosmic", "eta", "t_superconform", "H", "H_conf",
    "Omega_m", "Omega_r", "Omega_de_def", "w_de_def", "w_phi",
]

# <project root>/runs — resolved relative to this file, independent of cwd.
# Deliberately outside kmouflage/ (not package source/data) and outside
# examples/ (run_solver is core package functionality, not example-only —
# future callers like an MCMC likelihood will use the same cache).
DEFAULT_RUNS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"
)

INDEX_COLUMNS = [
    "key", "name", "path", "model", "coupling", "potential",
    "calibrated", "calibrate_target", "Omega_m0", "Omega_r0", "H0_input",
    "M4_tilde", "timestamp",
]


def save_run(bg: KMouflageBackground, name: str, outdir: str = DEFAULT_RUNS_DIR, fields=None, **extra_meta) -> str:
    """
    Sauvegarde un run KMouflageBackground déjà résolu (bg.run() appelé) :
      - param.ini : paramètres cosmo / model / coupling / potentiel / numériques
      - data.npz  : tableaux des quantités dérivées sur la grille N du solveur

    Retourne le chemin du dossier créé (outdir/name).
    """
    if not hasattr(bg, "_N"):
        raise RuntimeError("bg.run() doit être appelé avant save_run().")

    run_dir = os.path.join(outdir, name)
    os.makedirs(run_dir, exist_ok=True)

    config = configparser.ConfigParser()

    config["model"]     = {"name": bg.model.name,     **{k: str(v) for k, v in bg.model.params.items()}}
    config["coupling"]  = {"name": bg.coupling.name,  **{k: str(v) for k, v in bg.coupling.params.items()}}
    config["potential"] = {"name": bg.potential.name, **{k: str(v) for k, v in bg.potential.params.items()}}
    config["cosmology"] = {
        "Omega_m0":   str(bg.cosmo.Omega_m0),
        "Omega_r0":   str(bg.cosmo.Omega_r0),
        "M4_tilde":   str(bg.M4_tilde),
        "H0_input":   str(bg.cosmo.H0_input),
        "M_Pl_input": str(bg.cosmo.M_Pl_input),
    }
    config["initial_conditions"] = {
        "z_ini":   str(bg.ic.z_ini),
        "phi_ini": str(bg.ic.phi_ini),
    }
    config["numerics"] = {
        "rtol":     str(bg.rtol),
        "atol":     str(bg.atol),
        "max_step": str(bg.max_step),
        "N_points": str(bg.N_points),
    }
    config["results"] = {
        "M_Pl0_eff": str(bg.M_Pl0_eff),
        "delta_Mpl": str(bg.delta_Mpl),
        "delta_H":   str(bg.delta_H),
    }
    if extra_meta:
        config["meta"] = {k: str(v) for k, v in extra_meta.items()}

    with open(os.path.join(run_dir, "param.ini"), "w") as f:
        config.write(f)

    fields = fields or DEFAULT_FIELDS
    fields = list(dict.fromkeys([*MANDATORY_FIELDS, *fields]))  # a/phi/phi_prime always saved
    N_arr = bg._N
    data = {"N": N_arr}
    for field in fields:
        interp = getattr(bg, field, None)
        if interp is None:
            continue
        data[field] = interp(N_arr)
    np.savez(os.path.join(run_dir, "data.npz"), **data)

    return run_dir


def load_run(run_dir: str) -> dict:
    """
    Recharge un run sauvegardé par save_run().

    Retourne {"params": configparser.ConfigParser, "data": dict[str, np.ndarray],
    "run_dir": str}.
    """
    config = configparser.ConfigParser()
    config.read(os.path.join(run_dir, "param.ini"))

    npz = np.load(os.path.join(run_dir, "data.npz"))
    data = {key: npz[key] for key in npz.files}

    return {"params": config, "data": data, "run_dir": run_dir}


def _canonical_key(bg: KMouflageBackground, calibrate_target: float | None) -> str:
    """
    Hash of every parameter that affects the physics output of bg, so two
    calls with the same configuration produce the same key regardless of
    dict/attribute ordering. If calibrate_target is set, M4_tilde is an
    *output* of the run (not part of the key) — calibrate_target is the
    input instead.
    """
    payload = {
        "model_name":       bg.model.name,
        "model_params":     sorted(bg.model.params.items()),
        "coupling_name":    bg.coupling.name,
        "coupling_params":  sorted(bg.coupling.params.items()),
        "potential_name":   bg.potential.name,
        "potential_params": sorted(bg.potential.params.items()),
        "Omega_m0":   bg.cosmo.Omega_m0,
        "Omega_r0":   bg.cosmo.Omega_r0,
        "H0_input":   bg.cosmo.H0_input,
        "M_Pl_input": bg.cosmo.M_Pl_input,
        "z_ini":   bg.ic.z_ini,
        "phi_ini": bg.ic.phi_ini,
        "rtol": bg.rtol, "atol": bg.atol, "max_step": bg.max_step, "N_points": bg.N_points,
        "calibrate_target": calibrate_target,
        "M4_tilde": None if calibrate_target is not None else bg.M4_tilde,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _upsert_index(outdir: str, key: str, run_dir: str, bg: KMouflageBackground,
                   calibrate_target: float | None) -> None:
    """
    Write the index.csv row for `key`: appends it if the key isn't present
    yet, otherwise replaces the existing row in place. The replace path is
    what lets run_solver(..., overwrite=True) re-run a config without
    leaving a stale duplicate entry behind.
    """
    os.makedirs(outdir, exist_ok=True)
    index_path = os.path.join(outdir, "index.csv")

    row = {
        "key":              key,
        "name":             os.path.basename(run_dir),
        "path":             run_dir,
        "model":            bg.model.name,
        "coupling":         bg.coupling.name,
        "potential":        bg.potential.name,
        "calibrated":       calibrate_target is not None,
        "calibrate_target": calibrate_target,
        "Omega_m0":         bg.cosmo.Omega_m0,
        "Omega_r0":         bg.cosmo.Omega_r0,
        "H0_input":         bg.cosmo.H0_input,
        "M4_tilde":         bg.M4_tilde,
        "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    rows = []
    replaced = False
    if os.path.exists(index_path):
        with open(index_path, newline="") as f:
            for existing in csv.DictReader(f):
                if existing["key"] == key:
                    rows.append(row)
                    replaced = True
                else:
                    rows.append(existing)
    if not replaced:
        rows.append(row)

    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _next_run_name(outdir: str) -> str:
    """
    Simple sequential name run_0001, run_0002, ... — based on the highest
    existing run_<N> folder in outdir, so it stays correct even if some
    runs were deleted or the index.csv was hand-edited.
    """
    n = 0
    if os.path.isdir(outdir):
        for entry in os.listdir(outdir):
            m = re.fullmatch(r"run_(\d+)", entry)
            if m:
                n = max(n, int(m.group(1)))
    return f"run_{n + 1:04d}"


def _lookup_index(outdir: str, key: str) -> str | None:
    index_path = os.path.join(outdir, "index.csv")
    if not os.path.exists(index_path):
        return None
    with open(index_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["key"] == key:
                return row["path"]
    return None


def find_runs(outdir: str = DEFAULT_RUNS_DIR, **filters) -> list[dict]:
    """
    Search runs/index.csv for rows matching every given column=value filter
    (compared as strings). Returns the matching rows (list of dict).

    Example
    -------
    >>> find_runs(model="arctan (K_star=1000, X_star=100)", calibrated="True")
    """
    index_path = os.path.join(outdir, "index.csv")
    if not os.path.exists(index_path):
        return []
    with open(index_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in filters.items())]


def run_solver(
    model,
    coupling,
    cosmo=None,
    ic=None,
    potential=None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = 0.005,
    N_points: int = 100_000,
    calibrate: bool = False,
    outdir: str = DEFAULT_RUNS_DIR,
    overwrite: bool = False,
    fields=None,
    verbose: bool = True,
) -> dict:
    """
    Run (or load from cache) this exact configuration (model, coupling,
    cosmo, potential, ic, numerics, calibration flag) and return its data
    — computing and saving it only if runs/index.csv doesn't already have a
    matching entry.

    On a cache hit, nothing is re-simulated. Returns
    {"params": ConfigParser, "data": dict[str, ndarray], "run_dir": str}.

    calibrate=True tunes M4_tilde (via calibrate_M4_tilde) so that
    Omega_DE(z=0) matches 1 - Omega_m0 - Omega_r0 for this cosmo — the only
    target ever used in practice, since Omega_r0 is fixed and Omega_m0 is
    already part of cosmo — instead of leaving M4_tilde at
    CosmologicalParams' own GR-approximation default.

    overwrite=True forces a re-run even on a cache hit, and reuses the same
    run_dir / index.csv row (no duplicate entry) instead of creating a new
    one — use it when a run was saved before a bugfix/param tweak and needs
    refreshing in place.

    fields selects which derived quantities get saved (forwarded to
    save_run(); defaults to DEFAULT_FIELDS if omitted). Note that fields is
    *not* part of the cache key: it has no effect on a cache hit — pass
    overwrite=True too if a config is already cached under a different
    field list and you need it re-saved with this one.
    """
    bg = KMouflageBackground(
        model=model, coupling=coupling, cosmo=cosmo, ic=ic, potential=potential,
        rtol=rtol, atol=atol, max_step=max_step, N_points=N_points,
    )
    calibrate_target = (1.0 - bg.cosmo.Omega_m0 - bg.cosmo.Omega_r0) if calibrate else None
    key = _canonical_key(bg, calibrate_target)

    cached = _lookup_index(outdir, key)
    if cached is not None and os.path.isdir(cached) and not overwrite:
        if verbose:
            print(f"[run_solver] cache hit  (key={key}) -> {cached}")
        return load_run(cached)

    if verbose:
        reason = "overwrite requested" if cached is not None else "cache miss"
        print(f"[run_solver] {reason} (key={key}) -> running")
    if calibrate_target is not None:
        calibrate_M4_tilde(bg, target_Omega_DE=calibrate_target, verbose=False)
    else:
        bg.run(verbose=verbose)

    run_name = os.path.basename(cached) if cached is not None else _next_run_name(outdir)
    run_dir  = save_run(bg, run_name, outdir=outdir, fields=fields)
    _upsert_index(outdir, key, run_dir, bg, calibrate_target)
    return load_run(run_dir)
