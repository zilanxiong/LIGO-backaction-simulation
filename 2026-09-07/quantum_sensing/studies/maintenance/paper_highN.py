import sys, time; sys.path.insert(0, "2026-09-07")
import numpy as np, pandas as pd
import quantum_sensing as qs
qs.set_data_dir("2026-09-07/quantum_sensing/data")
from quantum_sensing import probes
from quantum_sensing.channels import get_state_ba1
from quantum_sensing.convergence import converged_qfi

R = "2026-09-07/quantum_sensing/studies/results"
path = f"{R}/ba1_frequency_sweep_paper_pn_ba1.csv"
df = pd.read_csv(path)
fams = {name: fam(5.0) for name, fam in probes.PROBE_FAMILIES.items()}
fams["opt_fock_sup"] = probes.optimized(5.0, "fock_sup")

todo = df[~df.converged].sort_values("f_hz", ascending=False)
for _, row in todo.iterrows():
    t0 = time.time()
    res = converged_qfi(fams[row.state], dynamics=get_state_ba1,
                        param_type="epsilon_a", kappa_ba=row.kappa_ba,
                        pn_in=0.2, eta_out=0.95,
                        N_start=700, N_step=200, N_max=1800)
    m = (df.state == row.state) & np.isclose(df.f_hz, row.f_hz, rtol=1e-9)
    df.loc[m, ["qfi", "N_basis", "converged"]] = [res["qfi"], res["N_basis"],
                                                  res["converged"]]
    df.to_csv(path, index=False)
    print(f"{row.state:13s} f={row.f_hz:6.2f} QFI={res['qfi']:.4f} "
          f"N={res['N_basis']} conv={res['converged']} "
          f"({time.time()-t0:.0f}s)", flush=True)
