import sys; sys.path.insert(0, "2026-09-07")
import numpy as np, pandas as pd
import quantum_sensing as qs
from quantum_sensing import gaussian as g

R = "2026-09-07/quantum_sensing/studies/results"
r = np.arcsinh(np.sqrt(2.0))
cfgs = {"": (dict(eta_out=0.9), "after"),
        "_ba2_detection": (dict(eta_out=0.9), "before"),
        "_ba3_full": (dict(eta_in=0.95, eta_ch=0.99, eta_out=0.90),
                      "simultaneous")}
for suf, (kw, so) in cfgs.items():
    path = f"{R}/ba1_frequency_sweep{suf}.csv"
    df = pd.read_csv(path)
    n_patch = 0
    for i, row in df[df.state == "sqz_vac_p"].iterrows():
        V, d = g.squeezed_state(-r)
        F = g.gaussian_qfi_rp(V, d, param_type="epsilon_a",
                              kappa_ba=row.kappa_ba, signal_order=so, **kw)
        if abs(row.qfi - F) / F > 1e-3:
            # Fock value truncation-corrupted (flagged unconverged);
            # replace with the exact Gaussian value. N_basis=0 marks
            # an analytic covariance-matrix entry.
            df.loc[i, ["qfi", "converged", "N_basis"]] = [F, True, 0]
            n_patch += 1
    df.to_csv(path, index=False)
    print(f"{suf or '_detection'}: patched {n_patch} sqz_vac_p rows with exact Gaussian")
