import sys; sys.path.insert(0, "2026-09-07")
import numpy as np, pandas as pd
import quantum_sensing as qs
from quantum_sensing import probes
from quantum_sensing.channels import get_state_ba1
from quantum_sensing.convergence import converged_qfi

R = "2026-09-07/quantum_sensing/studies/results"
path = f"{R}/ba1_frequency_sweep_pn_ba1_detection.csv"
factory = probes.squeezed_vacuum_p(2.0)
df = pd.read_csv(path)
for f_hz in [14.677992676220699, 10.0]:
    kappa = qs.rp_kappa(2 * np.pi * f_hz)
    res = converged_qfi(factory, dynamics=get_state_ba1, param_type="epsilon_a",
                        kappa_ba=kappa, pn_in=0.1, eta_out=0.9,
                        N_start=700, N_step=150, N_max=1700)
    print(f"f={f_hz:.2f} kappa={kappa:.3f} QFI={res['qfi']:.5f} "
          f"N={res['N_basis']} conv={res['converged']}", flush=True)
    m = (df.state == "sqz_vac_p") & (np.isclose(df.f_hz, f_hz, rtol=1e-6))
    df.loc[m, ["qfi", "N_basis", "converged"]] = [res["qfi"], res["N_basis"],
                                                  res["converged"]]
df.to_csv(path, index=False)
print("patched", path)
