"""
kmouflage/io_utils.py
=======================
Save / reload a KMouflageBackground run as a human-readable param.ini
(cosmology, model, coupling, potential, numerics) plus a data.npz of the
derived quantities on the solver's N grid, for reproducible comparisons
without re-integrating each time. Saved by default under <project root>/examples/runs/.
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

DEFAULT_FIELDS = [
    "z", "a", "t_cosmic", "eta", "t_superconform", "H", "H_conf",
    "Omega_m", "Omega_r", "Omega_de_def", "w_de_def", "w_phi",
]

# <project root>/examples/runs — resolved relative to this file, independent of cwd
DEFAULT_RUNS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "runs"
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

    Retourne {"params": configparser.ConfigParser, "data": dict[str, np.ndarray]}.
    """
    config = configparser.ConfigParser()
    config.read(os.path.join(run_dir, "param.ini"))

    npz = np.load(os.path.join(run_dir, "data.npz"))
    data = {key: npz[key] for key in npz.files}

    return {"params": config, "data": data}


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


def _append_index(outdir: str, key: str, run_dir: str, bg: KMouflageBackground,
                   calibrate_target: float | None) -> None:
    os.makedirs(outdir, exist_ok=True)
    index_path = os.path.join(outdir, "index.csv")
    write_header = not os.path.exists(index_path)

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
    with open(index_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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


def get_or_run(
    model,
    coupling,
    cosmo=None,
    ic=None,
    potential=None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = 0.005,
    N_points: int = 100_000,
    calibrate_target: float | None = None,
    calibrate_method: str = "brentq",
    outdir: str = DEFAULT_RUNS_DIR,
    verbose: bool = True,
) -> str:
    """
    Return the run directory for this exact configuration (model, coupling,
    cosmo, potential, ic, numerics, calibration target) — computing and
    saving it only if runs/index.csv doesn't already have a matching entry.

    On a cache hit, nothing is re-simulated: the existing run_dir is
    returned directly. Load its data with load_run(run_dir).
    """
    bg = KMouflageBackground(
        model=model, coupling=coupling, cosmo=cosmo, ic=ic, potential=potential,
        rtol=rtol, atol=atol, max_step=max_step, N_points=N_points,
    )
    key = _canonical_key(bg, calibrate_target)

    cached = _lookup_index(outdir, key)
    if cached is not None and os.path.isdir(cached):
        if verbose:
            print(f"[get_or_run] cache hit  (key={key}) -> {cached}")
        return cached

    if verbose:
        print(f"[get_or_run] cache miss (key={key}) -> running")
    bg.run(verbose=verbose)
    if calibrate_target is not None:
        calibrate_M4_tilde(bg, target_Omega_DE=calibrate_target, verbose=False, method=calibrate_method)

    run_dir = save_run(bg, _next_run_name(outdir), outdir=outdir)
    _append_index(outdir, key, run_dir, bg, calibrate_target)
    return run_dir
