"""
Unified helper for loading quantum state optimization results and reconstructing
QuTiP quantum states for fock_sup, sqz_vac, and sqz_coh state types.

Data directory must be configured before using load functions:
    import quantum_sensing
    quantum_sensing.set_data_dir("/path/to/consolidated_data")

Or via environment variable:
    export QUANTUM_SENSING_DATA_DIR=/path/to/consolidated_data
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import qutip as qt

# ---- Data directory configuration ----------------------------------------

_DATA_DIR = None


def set_data_dir(path):
    """Set the path to the consolidated_data directory."""
    global _DATA_DIR
    _DATA_DIR = Path(path)


def _get_data_dir():
    global _DATA_DIR
    if _DATA_DIR is not None:
        return _DATA_DIR
    env = os.environ.get("QUANTUM_SENSING_DATA_DIR")
    if env:
        return Path(env)
    raise RuntimeError(
        "Data directory not configured. Either:\n"
        "  - Call quantum_sensing.set_data_dir('/path/to/consolidated_data')\n"
        "  - Set QUANTUM_SENSING_DATA_DIR environment variable"
    )


# ---- Style constants (import * gives these to notebooks) -----------------

STATE_COLORS = {
    "fock_sup": "#1f77b4",
    "sqz_vac":  "#d62728",
    "sqz_coh":  "#2ca02c",
}
STATE_LABELS = {
    "fock_sup": "Fock Sup.",
    "sqz_vac":  "Sqz. Vac. Sup.",
    "sqz_coh":  "Sqz. Coh. Sup.",
}
STATE_MARKERS = {
    "fock_sup": "o",
    "sqz_vac":  "s",
    "sqz_coh":  "^",
}

# ---- Data loading --------------------------------------------------------

def load_results_df() -> pd.DataFrame:
    """Load master_results.csv, excluding rows marked valid=false."""
    csv_path = _get_data_dir() / "master_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"master_results.csv not found at {csv_path}\n"
            "Run rebuild_master_csv() from Julia to generate it."
        )
    df = pd.read_csv(csv_path, dtype={"valid": str})
    return df[df["valid"].fillna("").str.lower() != "false"].reset_index(drop=True)


def load_state_params(state_type: str, loss_config: str,
                      eta: float, pn: float) -> list[dict]:
    """
    Load all result dicts for one (state_type, loss_config, eta, pn) slice
    from states.h5.

    Returns a list of dicts with keys:
        state_type, loss_config, eta, pn,
        N_target, n_sups, N_basis, qfi, N_actual,
        c_vals, alpha_vals, r_vals, n_vals   (complex/int numpy arrays)
    """
    h5_path = _get_data_dir() / "states.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"states.h5 not found at {h5_path}\n"
            "Run: julia --project=julia_code consolidated_data/scripts/rebuild_all.jl"
        )

    eta_str = _fmt(eta)
    pn_str  = _fmt(pn)
    group_key  = f"{state_type}_{loss_config}"
    eta_pn_key = f"eta{eta_str}_pn{pn_str}"

    results = []
    with h5py.File(h5_path, "r") as h5f:
        if group_key not in h5f:
            return results
        grp = h5f[group_key]
        if eta_pn_key not in grp:
            return results
        eta_grp = grp[eta_pn_key]

        for entry_key, entry in eta_grp.items():
            attrs = dict(entry.attrs)
            c_re = entry["c_vals_re"][()]
            c_im = entry["c_vals_im"][()]
            a_re = entry["alpha_re"][()]
            a_im = entry["alpha_im"][()]
            r_re = entry["r_re"][()]
            r_im = entry["r_im"][()]
            nv   = entry["n_vals"][()].astype(int)

            results.append({
                "state_type":  state_type,
                "loss_config": loss_config,
                "eta":         float(eta),
                "pn":          float(pn),
                "N_target":    float(attrs.get("N_target", np.nan)),
                "n_sups":      int(attrs.get("n_sups", 0)),
                "N_basis":     int(attrs.get("N_basis", 0)),
                "qfi":         float(attrs.get("qfi", np.nan)),
                "N_actual":    float(attrs.get("N_actual", np.nan)),
                "c_vals":      c_re + 1j * c_im,
                "alpha_vals":  a_re + 1j * a_im,
                "r_vals":      r_re + 1j * r_im,
                "n_vals":      nv,
            })

    return results


def list_available_eta_pn(state_type: str, loss_config: str) -> list[str]:
    """List available eta_pn keys in states.h5 for a given (state_type, loss_config)."""
    h5_path = _get_data_dir() / "states.h5"
    if not h5_path.exists():
        return []
    group_key = f"{state_type}_{loss_config}"
    with h5py.File(h5_path, "r") as h5f:
        if group_key not in h5f:
            return []
        return sorted(h5f[group_key].keys())

# ---- Trend queries (DataFrame ops) --------------------------------------

def _filter(df: pd.DataFrame, state_type: str, loss_config: str,
            eta: float = None, pn: float = None,
            N_target: float = None, n_sups: int = None,
            tol: float = 1e-6) -> pd.DataFrame:
    mask = (df["state_type"] == state_type) & (df["loss_config"] == loss_config)
    if eta is not None:
        mask &= (df["eta"] - eta).abs() < tol
    if pn is not None:
        mask &= (df["pn"] - pn).abs() < tol
    if N_target is not None:
        mask &= (df["N_target"] - N_target).abs() < 0.01
    if n_sups is not None:
        mask &= (df["n_sups"] == n_sups)
    return df[mask]


def get_best_qfi_envelope(df: pd.DataFrame, state_type: str, loss_config: str,
                          eta: float, pn: float):
    """Max QFI over n_sups per N_target. Returns (N_arr, qfi_arr) or None."""
    sub = _filter(df, state_type, loss_config, eta=eta, pn=pn)
    sub = sub[sub["qfi"] > 0].dropna(subset=["qfi"])
    if sub.empty:
        return None
    best = sub.groupby("N_target")["qfi"].max().reset_index().sort_values("N_target")
    return best["N_target"].to_numpy(), best["qfi"].to_numpy()


def get_nsup_trend(df: pd.DataFrame, state_type: str, loss_config: str,
                   eta: float, pn: float, N_target: float):
    """QFI vs n_sups at fixed (eta, pn, N_target). Returns (ns_arr, qfi_arr) or None."""
    sub = _filter(df, state_type, loss_config, eta=eta, pn=pn, N_target=N_target)
    sub = sub[sub["qfi"] > 0].dropna(subset=["qfi"])
    if sub.empty:
        return None
    sub = sub.sort_values("n_sups")
    return sub["n_sups"].to_numpy(), sub["qfi"].to_numpy()


def get_eta_trend(df: pd.DataFrame, state_type: str, loss_config: str,
                  pn: float, N_target: float):
    """Best QFI vs eta at fixed (pn, N_target). Returns (eta_arr, qfi_arr) or None."""
    sub = _filter(df, state_type, loss_config, pn=pn, N_target=N_target)
    sub = sub[sub["qfi"] > 0].dropna(subset=["qfi"])
    if sub.empty:
        return None
    best = sub.groupby("eta")["qfi"].max().reset_index().sort_values("eta")
    return best["eta"].to_numpy(), best["qfi"].to_numpy()


def get_pn_trend(df: pd.DataFrame, state_type: str, loss_config: str,
                 eta: float, N_target: float):
    """Best QFI vs pn at fixed (eta, N_target). Returns (pn_arr, qfi_arr) or None."""
    sub = _filter(df, state_type, loss_config, eta=eta, N_target=N_target)
    sub = sub[sub["qfi"] > 0].dropna(subset=["qfi"])
    if sub.empty:
        return None
    best = sub.groupby("pn")["qfi"].max().reset_index().sort_values("pn")
    return best["pn"].to_numpy(), best["qfi"].to_numpy()

# ---- Wavefunction utilities ----------------------------------------------

def harmonic_oscillator_wavefunction(n, q):
    """
    n-th harmonic oscillator wavefunction phi_n(q) evaluated at positions q.

    Uses the stable recurrence on full wavefunctions (not raw Hermite polynomials):
        phi_0(q) = pi^{-1/4} exp(-q^2/2)
        phi_{n+1}(q) = sqrt(2/(n+1)) q phi_n(q) - sqrt(n/(n+1)) phi_{n-1}(q)
    """
    q = np.asarray(q, dtype=float)
    gauss = np.exp(-q**2 / 2.0)
    phi_prev = (np.pi ** (-0.25)) * gauss  # phi_0

    if n == 0:
        return phi_prev

    phi_curr = np.sqrt(2.0) * q * phi_prev  # phi_1

    for k in range(1, n):
        phi_next = np.sqrt(2.0 / (k + 1)) * q * phi_curr - np.sqrt(k / (k + 1)) * phi_prev
        phi_prev = phi_curr
        phi_curr = phi_next

    return phi_curr


def _ho_basis_matrix(q, N_basis):
    """
    Build (N_basis, len(q)) matrix of HO wavefunctions via single-pass recurrence.
    Row n holds phi_n(q).
    """
    q = np.asarray(q, dtype=float)
    Nq = len(q)
    Phi = np.empty((N_basis, Nq), dtype=float)

    gauss = np.exp(-q**2 / 2.0)
    Phi[0] = (np.pi ** (-0.25)) * gauss

    if N_basis >= 2:
        Phi[1] = np.sqrt(2.0) * q * Phi[0]

    for n in range(2, N_basis):
        Phi[n] = np.sqrt(2.0 / n) * q * Phi[n-1] - np.sqrt((n-1) / n) * Phi[n-2]

    return Phi


def _simpson_weights(q):
    """Composite Simpson's rule weights. Falls back to trapezoidal for odd-length input."""
    q = np.asarray(q, dtype=float)
    N = len(q)
    w = np.zeros(N)

    if N < 3 or N % 2 == 0:
        # Trapezoidal fallback (N is even means odd number of intervals)
        dq = np.diff(q)
        w[0] = dq[0] / 2
        w[1:-1] = (dq[:-1] + dq[1:]) / 2
        w[-1] = dq[-1] / 2
        return w

    # Composite Simpson on pairs of intervals
    for i in range(0, N - 2, 2):
        h = q[i+2] - q[i]
        h1 = q[i+1] - q[i]
        h2 = q[i+2] - q[i+1]
        w[i]   += h * (2*h1 - h2) / (6*h1)
        w[i+1] += h**3 / (6*h1*h2)
        w[i+2] += h * (2*h2 - h1) / (6*h2)

    return w


def wavefunction_to_qobj(q, psi, N_basis=25):
    """
    Convert a position-space wavefunction to a QuTiP Qobj in the Fock basis.

    Computes all N_basis overlaps <n|psi> in a single pass using the HO
    three-term recurrence and Simpson's rule integration.

    Args:
        q: 1D array of position coordinates.
        psi: 1D array of wavefunction values (can be complex).
        N_basis: Hilbert space dimension (Fock basis cutoff).

    Returns:
        qt.Qobj: Normalized ket in the Fock basis.
    """
    q = np.asarray(q, dtype=float)
    psi = np.asarray(psi, dtype=complex)

    w = _simpson_weights(q)

    # Normalize input
    norm2 = np.sum(w * np.abs(psi)**2)
    psi_norm = psi / np.sqrt(norm2)

    # Weighted wavefunction
    w_psi = w * psi_norm

    # Build basis matrix and compute all overlaps via matrix-vector product
    Phi = _ho_basis_matrix(q, N_basis)
    coeffs = Phi @ w_psi  # (N_basis, Nq) @ (Nq,) -> (N_basis,)

    state_qobj = qt.Qobj(coeffs, dims=[[N_basis], [1]])
    return state_qobj.unit()


# ---- State reconstruction ------------------------------------------------

def reconstruct_fock_sup(n_vals, c_vals, N_basis: int) -> qt.Qobj:
    """Build |psi> = sum_k c_k |n_k> and normalise."""
    psi = qt.Qobj(np.zeros(N_basis, dtype=complex))
    for n, c in zip(n_vals, c_vals):
        n = int(n)
        if 0 <= n < N_basis:
            psi += c * qt.basis(N_basis, n)
    return psi.unit()


def reconstruct_sqz_vac(c_vals, r_vals, N_basis: int) -> qt.Qobj:
    """Build |psi> = sum_k c_k S(r_k)|0> and normalise."""
    vac = qt.basis(N_basis, 0)
    psi = qt.Qobj(np.zeros(N_basis, dtype=complex))
    for c, r in zip(c_vals, r_vals):
        if abs(c) < 1e-14:
            continue
        sq = qt.squeeze(N_basis, complex(r)) * vac if abs(r) > 1e-12 else vac
        psi += c * sq
    return psi.unit()


def reconstruct_sqz_coh(c_vals, alpha_vals, r_vals, N_basis: int) -> qt.Qobj:
    """Build |psi> = sum_k c_k D(alpha_k) S(r_k)|0> and normalise."""
    vac = qt.basis(N_basis, 0)
    psi = qt.Qobj(np.zeros(N_basis, dtype=complex))
    for c, alpha, r in zip(c_vals, alpha_vals, r_vals):
        if abs(c) < 1e-14:
            continue
        ket = vac
        if abs(r) > 1e-12:
            ket = qt.squeeze(N_basis, complex(r)) * ket
        if abs(alpha) > 1e-12:
            ket = qt.displace(N_basis, complex(alpha)) * ket
        psi += c * ket
    return psi.unit()


def reconstruct_state(result: dict, N_basis: int = None) -> qt.Qobj:
    """Dispatch state reconstruction based on result['state_type'].

    Parameters
    ----------
    result  : dict returned by load_state_params()
    N_basis : override the stored Hilbert-space dimension (e.g. to up-sample
              or down-sample). Defaults to result['N_basis'].
    """
    st     = result["state_type"]
    N      = N_basis if N_basis is not None else result["N_basis"]
    c_vals = result["c_vals"]

    if st == "fock_sup":
        return reconstruct_fock_sup(result["n_vals"], c_vals, N)
    elif st == "sqz_vac":
        return reconstruct_sqz_vac(c_vals, result["r_vals"], N)
    elif st in ("sqz_coh", "sqz_cat"):
        return reconstruct_sqz_coh(c_vals, result["alpha_vals"], result["r_vals"], N)
    else:
        raise ValueError(f"Unknown state_type: {st!r}")


def get_wigner(result: dict, xrange: tuple = (-5.0, 5.0), pts: int = 200,
               N_basis: int = None):
    """
    Reconstruct state from result dict and compute Wigner function.

    Parameters
    ----------
    N_basis : override Hilbert-space dimension (passed to reconstruct_state).

    Returns (xvec, W) where W has shape (pts, pts).
    """
    psi  = reconstruct_state(result, N_basis=N_basis)
    xvec = np.linspace(xrange[0], xrange[1], pts)
    W    = qt.wigner(psi, xvec, xvec)
    return xvec, W

# ---- Internal helpers ----------------------------------------------------

def _fmt(v: float) -> str:
    """Replicate Julia's fmt(): %.10g with dots replaced by 'p'."""
    s = f"{float(v):.10g}"
    return s.replace(".", "p")
