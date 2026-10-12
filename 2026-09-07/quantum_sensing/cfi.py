"""
Classical Fisher Information for various measurement types.

Python port of julia/src/cfi.jl with identical physics and constants.

Every measurement reduces to the same formula on its outcome distribution:

    CFI = sum_m (d_theta p_m)^2 / p_m,   p_m = Tr[rho Pi_m],
                                         d_theta p_m = Tr[drho Pi_m]

with (rho, drho = d rho / d theta) either supplied directly (e.g. from a
cache) or built by central finite differences through a `dynamics` callable
(same pattern as :func:`quantum_sensing.sld.calculate_qfi`).

Measurement types
-----------------
photon_counting : Fock projectors (diagonal of rho, drho)
homodyne        : eigenbasis of X_theta = (a e^{-i theta} + a^dag e^{i theta})/sqrt(2)
heterodyne      : coherent-state POVM |alpha><alpha|/pi on a phase-space grid
sld_optimal     : eigenbasis of the SLD (saturates the QFI by construction)
projective      : arbitrary user-supplied projector vectors
"""

import warnings

import numpy as np
from numpy.linalg import eigh
import qutip as qt

PREC_DIFF_CFI = 1e-5
PROB_FLOOR = 1e-15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_rho_drho(dynamics, param_value=0.0, param_type="Delta",
                  prec=PREC_DIFF_CFI, **kwargs):
    """(rho, drho) by central finite differences through `dynamics`."""
    def dense_dm(params):
        state = dynamics(**params)
        if state.isket:
            v = state.full()
            return v @ v.conj().T
        return state.full()

    rho_plus = dense_dm({**kwargs, param_type: param_value + prec})
    rho_minus = dense_dm({**kwargs, param_type: param_value - prec})
    rho = dense_dm({**kwargs, param_type: param_value})
    drho = (rho_plus - rho_minus) / (2 * prec)
    return rho, drho


def _cfi_from_probs(probs, dprobs, prob_floor=PROB_FLOOR):
    """CFI = sum dp^2/p over outcomes with p > prob_floor."""
    probs = np.asarray(probs, dtype=float)
    dprobs = np.asarray(dprobs, dtype=float)
    mask = probs > prob_floor
    return float(np.sum(dprobs[mask] ** 2 / probs[mask]))


def _coherent_state_vec(alpha, N):
    """Fock-basis amplitudes of |alpha> up to N-1 (stable recurrence)."""
    v = np.empty(N, dtype=complex)
    v[0] = np.exp(-abs(alpha) ** 2 / 2)
    for n in range(1, N):
        v[n] = v[n - 1] * alpha / np.sqrt(n)
    return v


def _projector_probs(rho, drho, vecs):
    """p_m, dp_m for projectors onto the columns of `vecs`."""
    probs = np.real(np.einsum("im,ij,jm->m", vecs.conj(), rho, vecs, optimize=True))
    dprobs = np.real(np.einsum("im,ij,jm->m", vecs.conj(), drho, vecs, optimize=True))
    return probs, dprobs


# ---------------------------------------------------------------------------
# Measurement-specific CFI functions
# ---------------------------------------------------------------------------

def calculate_cfi_photon_counting(rho, drho):
    """CFI of ideal photon counting: p_m = rho_mm."""
    probs = np.real(np.diag(rho)).copy()
    dprobs = np.real(np.diag(drho)).copy()
    return {
        "cfi": _cfi_from_probs(probs, dprobs),
        "probabilities": probs,
        "dprobabilities": dprobs,
        "measurement_type": "photon_counting",
    }


def calculate_cfi_homodyne(rho, drho, theta_LO=0.0):
    """CFI of balanced homodyne at LO phase theta_LO, via the eigenbasis of
    the truncated quadrature operator X_theta = (a e^{-i th} + a^dag e^{i th})/sqrt(2).

    Convention: theta_LO = 0 measures X = (a + a^dag)/sqrt(2).
    """
    N = rho.shape[0]
    a = qt.destroy(N).full()
    X_theta = (a * np.exp(-1j * theta_LO) + a.conj().T * np.exp(1j * theta_LO)) / np.sqrt(2)
    _, evecs = eigh(X_theta)

    probs, dprobs = _projector_probs(rho, drho, evecs)
    probs = np.maximum(probs, 0.0)

    return {
        "cfi": _cfi_from_probs(probs, dprobs),
        "probabilities": probs,
        "dprobabilities": dprobs,
        "theta_LO": theta_LO,
        "measurement_type": "homodyne",
    }


def calculate_cfi_sld_optimal(rho, drho, sld_eigenvectors=None, prec_eig=1e-8):
    """CFI of the SLD-eigenbasis measurement (saturates the QFI).

    If `sld_eigenvectors` is given, evaluate that fixed basis instead
    (useful for testing a basis calibrated at a different channel point).
    Also returns the QFI.
    """
    N = rho.shape[0]
    evals_rho, evecs_rho = eigh(rho)
    drho_rot = evecs_rho.conj().T @ drho @ evecs_rho

    if sld_eigenvectors is None:
        denom = evals_rho[:, None] + evals_rho[None, :]
        L_rot = np.where(np.abs(denom) > prec_eig, 2.0 * drho_rot / np.where(denom == 0, 1, denom), 0.0)
        L = evecs_rho @ L_rot @ evecs_rho.conj().T
        L = (L + L.conj().T) / 2.0
        qfi = float(np.real(np.trace(rho @ L @ L)))
        sld_evals, sld_evecs = eigh(L)
    else:
        sld_evecs = sld_eigenvectors
        p_basis, dp_basis = _projector_probs(rho, drho, sld_evecs)
        sld_evals = dp_basis / np.maximum(p_basis, PROB_FLOOR)
        denom = evals_rho[:, None] + evals_rho[None, :]
        mask = np.abs(denom) > prec_eig
        qfi = float(np.real(np.sum(2.0 * np.abs(drho_rot[mask]) ** 2
                                   / denom[mask])))

    probs, dprobs = _projector_probs(rho, drho, sld_evecs)
    return {
        "cfi": _cfi_from_probs(probs, dprobs),
        "probabilities": probs,
        "dprobabilities": dprobs,
        "sld_eigenvalues": sld_evals,
        "sld_eigenvectors": sld_evecs,
        "qfi": qfi,
        "measurement_type": "sld_optimal",
    }


def calculate_cfi_projective(rho, drho, projectors):
    """CFI of an arbitrary projective measurement.

    `projectors` is a list of state vectors (1-D arrays) or a matrix whose
    columns are the projector vectors.
    """
    vecs = np.column_stack(projectors) if isinstance(projectors, (list, tuple)) \
        else np.asarray(projectors)
    probs, dprobs = _projector_probs(rho, drho, vecs)
    return {
        "cfi": _cfi_from_probs(probs, dprobs),
        "probabilities": probs,
        "dprobabilities": dprobs,
        "measurement_type": "projective",
    }


def calculate_cfi_heterodyne(rho, drho, n_grid=60, alpha_max=None, N_target=None):
    """CFI of ideal heterodyne (double homodyne): POVM |alpha><alpha|/pi on
    an n_grid x n_grid phase-space grid of half-width alpha_max.

    If alpha_max is None it is sized from N_target, or else from <n> of rho.
    Warns if the grid misses probability (normalization far from 1).
    """
    N = rho.shape[0]

    if alpha_max is None:
        if N_target is not None:
            alpha_max = max(3.0, 1.5 * np.sqrt(N_target + 1))
        else:
            n_bar = float(np.real(np.sum(np.arange(N) * np.real(np.diag(rho)))))
            alpha_max = max(3.0, 1.5 * np.sqrt(n_bar + 3 * np.sqrt(max(n_bar, 0.0)) + 1))

    alpha_re = np.linspace(-alpha_max, alpha_max, n_grid)
    alpha_im = np.linspace(-alpha_max, alpha_max, n_grid)
    dA = (alpha_re[1] - alpha_re[0]) * (alpha_im[1] - alpha_im[0])

    # coherent-state matrix: columns |alpha_k> for every grid point
    ar, ai = np.meshgrid(alpha_re, alpha_im)
    alphas = (ar + 1j * ai).ravel()
    ns = np.arange(N)
    # log-domain recurrence, vectorized: v_n = e^{-|a|^2/2} a^n / sqrt(n!)
    with np.errstate(divide="ignore"):
        log_fact = np.cumsum(np.log(np.maximum(ns, 1)))
    V = np.exp(-np.abs(alphas[None, :]) ** 2 / 2
               + ns[:, None] * np.log(np.where(alphas == 0, 1, alphas))[None, :]
               - log_fact[:, None] / 2)
    V[:, alphas == 0] = 0.0
    V[0, alphas == 0] = 1.0

    raw_probs = np.maximum(np.real(np.einsum("ik,ij,jk->k", V.conj(), rho, V,
                                             optimize=True)), 0.0)
    raw_dprobs = np.real(np.einsum("ik,ij,jk->k", V.conj(), drho, V, optimize=True))

    # F_C = int (dQ/dtheta)^2 / Q d^2alpha with Q(alpha) = <alpha|rho|alpha> / pi.
    # Per-cell prob is p = Q dA, so dp^2/p carries one net factor of 1/pi.
    # Matches julia/src/cfi.jl calculate_cfi_heterodyne.
    cfi = _cfi_from_probs(raw_probs, raw_dprobs) * dA / np.pi

    normalization = float(raw_probs.sum() * dA / np.pi)
    if abs(normalization - 1.0) > 0.05:
        warnings.warn(
            f"Heterodyne grid normalization = {normalization:.4f} "
            "(expected ~ 1.0). Consider increasing n_grid or alpha_max.")

    return {
        "cfi": cfi,
        "alpha_max": alpha_max,
        "n_grid": n_grid,
        "normalization": normalization,
        "measurement_type": "heterodyne",
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def calculate_cfi(dynamics=None, param_value=0.0, param_type="Delta",
                  prec=PREC_DIFF_CFI, rho=None, drho=None,
                  measurement="photon_counting", **kwargs):
    """Unified CFI: supply either (rho, drho) or a `dynamics` callable
    (finite-differenced in `param_type`, same pattern as calculate_qfi).

    measurement in {"photon_counting", "homodyne", "heterodyne",
                    "sld_optimal", "projective"}.
    Measurement options are passed through kwargs (theta_LO, n_grid,
    alpha_max, N_target, sld_eigenvectors, projectors); remaining kwargs go
    to `dynamics`.
    """
    meas_keys = {"theta_LO", "n_grid", "alpha_max", "N_target",
                 "sld_eigenvectors", "projectors"}
    meas_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in meas_keys}

    if rho is None or drho is None:
        if dynamics is None:
            raise ValueError("Must provide either `dynamics` or both `rho` and `drho`")
        if rho is not None:
            # `rho` without `drho` is the *input* probe state for `dynamics`
            # (same meaning as in calculate_qfi), not a precomputed output.
            kwargs["rho"] = rho
        rho, drho = _get_rho_drho(dynamics, param_value=param_value,
                                  param_type=param_type, prec=prec, **kwargs)

    if measurement == "photon_counting":
        return calculate_cfi_photon_counting(rho, drho)
    elif measurement == "homodyne":
        return calculate_cfi_homodyne(rho, drho,
                                      theta_LO=meas_kwargs.get("theta_LO", 0.0))
    elif measurement == "heterodyne":
        return calculate_cfi_heterodyne(rho, drho,
                                        n_grid=meas_kwargs.get("n_grid", 60),
                                        alpha_max=meas_kwargs.get("alpha_max"),
                                        N_target=meas_kwargs.get("N_target"))
    elif measurement == "sld_optimal":
        return calculate_cfi_sld_optimal(
            rho, drho, sld_eigenvectors=meas_kwargs.get("sld_eigenvectors"))
    elif measurement == "projective":
        return calculate_cfi_projective(rho, drho, meas_kwargs["projectors"])
    else:
        raise ValueError(
            f"Unknown measurement type: {measurement}. Use 'photon_counting', "
            "'homodyne', 'heterodyne', 'sld_optimal', or 'projective'")


# ---------------------------------------------------------------------------
# Compare multiple measurements
# ---------------------------------------------------------------------------

def compare_cfi(dynamics=None, rho=None, drho=None, param_value=0.0,
                param_type="Delta", prec=PREC_DIFF_CFI,
                measurements=("photon_counting", "homodyne", "sld_optimal"),
                homodyne_phases=(0.0,), **kwargs):
    """CFI of several measurements on the same (rho, drho), plus the QFI and
    per-measurement efficiency CFI/QFI. Homodyne is maximized over
    `homodyne_phases`.
    """
    meas_keys = {"n_grid", "alpha_max", "N_target", "projectors"}
    meas_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in meas_keys}

    if rho is None or drho is None:
        if dynamics is None:
            raise ValueError("Must provide either `dynamics` or both `rho` and `drho`")
        if rho is not None:
            # `rho` without `drho` is the *input* probe state for `dynamics`
            # (same meaning as in calculate_qfi), not a precomputed output.
            kwargs["rho"] = rho
        rho, drho = _get_rho_drho(dynamics, param_value=param_value,
                                  param_type=param_type, prec=prec, **kwargs)

    results = {}
    sld_result = calculate_cfi_sld_optimal(rho, drho)
    qfi = sld_result["qfi"]

    for meas in measurements:
        if meas == "photon_counting":
            results["photon_counting"] = calculate_cfi_photon_counting(rho, drho)
        elif meas == "homodyne":
            best = None
            for th in homodyne_phases:
                res = calculate_cfi_homodyne(rho, drho, theta_LO=th)
                if best is None or res["cfi"] > best["cfi"]:
                    best = res
            results["homodyne"] = best
        elif meas == "heterodyne":
            results["heterodyne"] = calculate_cfi_heterodyne(
                rho, drho,
                n_grid=meas_kwargs.get("n_grid", 60),
                alpha_max=meas_kwargs.get("alpha_max"),
                N_target=meas_kwargs.get("N_target"))
        elif meas == "sld_optimal":
            results["sld_optimal"] = sld_result
        elif meas == "projective":
            if "projectors" in meas_kwargs:
                results["projective"] = calculate_cfi_projective(
                    rho, drho, meas_kwargs["projectors"])

    best_meas, best_cfi = "sld_optimal", 0.0
    for k, v in results.items():
        if v["cfi"] > best_cfi:
            best_cfi, best_meas = v["cfi"], k

    efficiency = {k: (v["cfi"] / qfi if qfi > 0 else 0.0) for k, v in results.items()}

    return {
        "qfi": qfi,
        "results": results,
        "best_measurement": best_meas,
        "best_cfi": best_cfi,
        "efficiency": efficiency,
    }
