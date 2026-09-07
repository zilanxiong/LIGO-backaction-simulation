"""
Step 2 — how much does radiation-pressure back-action hurt the paper's
optimized states across the GW band, and does the loss placement matter?

Takes the optimized states from states.h5 (loss_ch, eta=0.9, pn=0,
N_target = 5, best n_sups per family) plus a coherent benchmark, and
evolves each through the back-action channel with the KLMTV-calibrated
frequency-dependent gain kappa_ba(Omega) = rp_kappa(Omega) (aLIGO O4
defaults), sweeping 10 Hz - 1 kHz for three loss placements:

    before : loss, then the back-action + signal stage   (eta_in)
    during : loss concurrent with back-action + signal   (eta_ch)
    after  : back-action + signal stage, then loss       (eta_out)

QFI is for the paper's sensing parameter epsilon_p.  The kappa_ba = 0
baseline of each placement is recorded alongside.

Numerics: the "during" case uses the package's mesolve channel
(get_state_single_mode_rp; tight tolerances).  For "before"/"after" the
back-action stage is loss-free, so the state is evolved by a dense matrix
exponential (exact) and the loss stage by the exact Kraus map of the
attenuator — cross-validated against the mesolve channel below the run.
Fock cutoffs are sized per (state, kappa_ba) by probing the sheared
state's support; a convergence check at the strongest shear is printed.

Usage:  python quantum_sensing/studies/step2_frequency_sweep.py
Output: quantum_sensing/studies/results/step2_sweep.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import qutip as qt
from scipy.linalg import expm
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import quantum_sensing as qs
from quantum_sensing.states import load_state_params, reconstruct_state

qs.set_data_dir(ROOT / "quantum_sensing" / "data")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# --- Study condition (one slice of the paper's grid) -----------------------
ETA = 0.9          # transmission of the loss stage
PN = 0.0           # phase noise
NT = 5.0           # photon-number budget
FREQS_HZ = np.logspace(1.0, 3.0, 11)          # 10 Hz - 1 kHz
PLACEMENTS = ["before", "during", "after"]
N_CAP = 600


def cutoff_for(kappa_ba):
    """Legacy heuristic cutoff (kept for step2b); superseded by probing."""
    return int(min(150, 60 + 18 * kappa_ba))


def load_best_states():
    """Best-QFI entry per state family at (loss_ch, ETA, PN, NT)."""
    states = {}
    for st in ["fock_sup", "sqz_vac", "sqz_coh"]:
        entries = [e for e in load_state_params(st, "loss_ch", ETA, PN)
                   if abs(e["N_target"] - NT) < 0.01]
        if entries:
            best = max(entries, key=lambda e: e["qfi"])
            states[st] = best
            print(f"{st}: n_sups={best['n_sups']}  paper QFI={best['qfi']:.4f}")
    return states


# --- Exact fast path for the loss-free back-action stage -------------------

def _ops(N):
    a = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)
    x2 = (a + a.conj().T) @ (a + a.conj().T) / 2
    Hsig = 1j * (a.conj().T - a)          # epsilon_p couples to i(a^dag - a)
    return x2, Hsig


def kraus_loss(rho, eta):
    """Exact attenuator channel (transmission eta) on a Fock density matrix."""
    if eta >= 1.0:
        return rho
    N = rho.shape[0]
    out = np.zeros_like(rho)
    for k in range(N):
        n = np.arange(k, N)
        logc = 0.5 * (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
                      + (n - k) * np.log(eta) + k * np.log(1 - eta))
        c = np.exp(logc)
        if c.max() < 1e-14:
            break
        out[:N - k, :N - k] += rho[k:, k:] * np.outer(c, c)
    return out


def make_fast_dynamics(build, N, g, eta, placement):
    """Dynamics callable for calculate_qfi: exact expm + Kraus loss."""
    x2, Hsig = _ops(N)
    v = build(N).full().ravel()

    def dyn(epsilon_p=0.0, **_):
        U = expm(-1j * (epsilon_p * Hsig + g * x2))
        if placement == "before":
            rho = kraus_loss(np.outer(v, v.conj()), eta)
            rho = U @ rho @ U.conj().T
        else:  # "after"
            w = U @ v
            rho = kraus_loss(np.outer(w, w.conj()), eta)
        return qt.Qobj(rho)

    return dyn


# --- Cutoff sizing ---------------------------------------------------------

def probe_cutoff(build, g, n_probe=N_CAP, tail=1e-9, floor=60):
    """Fock levels needed to hold the sheared state (loss-free = worst case)."""
    if g < 0.1:
        return floor
    x2, _ = _ops(n_probe)
    w = expm(-1j * g * x2) @ build(n_probe).full().ravel()
    pop = np.abs(w) ** 2
    cum_tail = np.cumsum(pop[::-1])[::-1]
    idx = np.argmax(cum_tail < tail)
    if idx == 0:          # tail never dropped below threshold within n_probe
        return n_probe
    return int(min(n_probe, max(floor, 1.25 * idx + 25)))


# --- QFI evaluation --------------------------------------------------------

def qfi_at(build, kappa_ba, placement, N):
    g = qs.ba_to_g(kappa_ba)
    if placement == "during":
        psi = build(N)
        return qs.calculate_qfi(qs.get_state_single_mode_rp,
                                param_type="epsilon_p", rho=psi, N_basis=N,
                                kappa_ba=kappa_ba, eta_ch=ETA, pn_ch=PN)
    dyn = make_fast_dynamics(build, N, g, ETA, placement)
    return qs.calculate_qfi(dyn, param_type="epsilon_p")


def crosscheck(build, kappa_ba, N):
    """Fast path vs the package mesolve channel, for before and after."""
    for placement, kw in [("before", {"eta_in": ETA}), ("after", {"eta_out": ETA})]:
        q_fast = qfi_at(build, kappa_ba, placement, N)
        psi = build(N)
        q_ref = qs.calculate_qfi(qs.get_state_single_mode_rp,
                                 param_type="epsilon_p", rho=psi, N_basis=N,
                                 kappa_ba=kappa_ba, pn_ch=PN, **kw)
        rel = abs(q_fast - q_ref) / q_ref
        print(f"  crosscheck {placement}: fast={q_fast:.6f} "
              f"mesolve={q_ref:.6f} rel={rel:.2e}")
        assert rel < 1e-5, "fast path disagrees with mesolve channel"


def main():
    best = load_best_states()
    builders = {st: (lambda N, e=e: reconstruct_state(e, N_basis=N))
                for st, e in best.items()}
    builders["coherent"] = lambda N: qt.coherent(N, np.sqrt(NT))

    print("\ncross-validating the exact fast path at kappa_ba ~ 1 ...")
    crosscheck(builders["sqz_vac"], 1.0, 100)

    kappas = qs.rp_kappa(2 * np.pi * FREQS_HZ)
    rows = []
    for name, build in builders.items():
        cuts = {kba: probe_cutoff(build, qs.ba_to_g(kba)) for kba in kappas}
        print(f"\n{name}: cutoffs "
              + " ".join(f"{c}" for c in cuts.values()))
        for placement in PLACEMENTS:
            q0 = qfi_at(build, 0.0, placement, 60)
            print(f" {placement}: no-BA baseline QFI = {q0:.4f}", flush=True)
            for f_hz, kba in zip(FREQS_HZ, kappas):
                t0 = time.time()
                N = cuts[kba]
                q = qfi_at(build, kba, placement, N)
                rows.append(dict(state=name, placement=placement,
                                 freq_hz=f_hz, kappa_ba=kba, qfi=q,
                                 qfi_noba=q0, N_basis=N))
                print(f"  f={f_hz:7.1f} Hz  kappa={kba:6.3f}  N={N:3d}  "
                      f"QFI={q:9.4f}  ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "step2_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    # Convergence check at the strongest shear (largest cutoff case)
    kmax = kappas.max()
    for name in ["fock_sup", "sqz_vac"]:
        if name not in builders:
            continue
        N0 = probe_cutoff(builders[name], qs.ba_to_g(kmax))
        N1 = min(N_CAP + 100, N0 + 80)
        q0 = df[(df.state == name) & (df.placement == "after")
                & (df.kappa_ba == kmax)].qfi.iloc[0]
        dyn = make_fast_dynamics(builders[name], N1, qs.ba_to_g(kmax), ETA, "after")
        q1 = qs.calculate_qfi(dyn, param_type="epsilon_p")
        print(f"convergence ({name}, after, kappa={kmax:.2f}): "
              f"N={N0} -> {q0:.5f},  N={N1} -> {q1:.5f}, "
              f"rel diff = {abs(q1-q0)/q1:.2e}")


if __name__ == "__main__":
    main()
