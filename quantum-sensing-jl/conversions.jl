# Unit conversions — canonical versions, shared by CPU and GPU code paths.

export loss_to_kappa, phirms_to_chi, qfi_to_phi_min
export dB_to_r, r_to_dB, var_to_dB, r_to_var
export dB_to_n, n_to_dB, r_to_n, n_to_r
export loss_lim_dB

loss_to_kappa(loss::Real, time::Real=1.0) = -log(1 - loss) / time
phirms_to_chi(phirms::Real, time::Real=1.0) = phirms^2 / time
qfi_to_phi_min(qfi::Real) = 1.0 / sqrt(qfi)

dB_to_r(dB::Real) = dB / (20 * log10(exp(1)))
r_to_dB(r::Real) = 20 * r * log10(exp(1))
var_to_dB(var::Real) = 10 * log10(var)
r_to_var(r::Real) = exp(-r^2)

dB_to_n(dB::Real) = sinh(dB_to_r(dB))^2
n_to_dB(n::Real) = r_to_dB(asinh(√n))
r_to_n(r::Real) = sinh(r)^2
n_to_r(n::Real) = asinh(√n)

loss_lim_dB(loss::Real) = -10 * log10(loss)
