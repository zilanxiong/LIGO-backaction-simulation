# Paper baseline reproduction (Ntarget = 5, N_basis = 40)

Channel: dephasing (prep) -> displacement signal + loss (sensing), as in Maliakal et al. Fig. 1.  `advantage = 10 log10(F_best/F_gauss)`; `F_best = max(F_gauss, F_fock_sup)`.

| eta | sigma_phi | F_gauss | F_fock_sup | adv (dB) | paper F_opt | paper adv | bound |
|---|---|---|---|---|---|---|---|
| 0.95 | 0.0 | 41.79 | 39.68 | 0.00 | 42.1 | 0.0 | 42.9 |
| 0.95 | 0.1 | 23.27 | 31.72 | 1.34 | 31.7 | 1.3 | 42.5 |
| 0.95 | 0.2 | 18.24 | 30.53 | 2.24 | 30.5 | 2.2 | 41.2 |
| 0.99 | 0.0 | 72.22 | 70.74 | 0.00 | 72.3 | 0.0 | 72.6 |
| 0.99 | 0.1 | 53.35 | 67.79 | 1.04 | 67.8 | 1.0 | 71.9 |
| 0.99 | 0.2 | 50.84 | 65.57 | 1.10 | 65.5 | 1.1 | 69.8 |
