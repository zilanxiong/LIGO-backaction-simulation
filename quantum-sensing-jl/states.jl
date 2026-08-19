# State parametrization: Fock superpositions and squeezed/coherent superpositions.

# ── Wavefunction utilities ────────────────────────────────────────────────

export harmonic_oscillator_wavefunction, wavefunction_to_quantumobject

"""
    harmonic_oscillator_wavefunction(n, q)

n-th harmonic oscillator wavefunction φ_n(q) evaluated at positions `q`.
"""
function harmonic_oscillator_wavefunction(n::Int, q::AbstractVector{<:Real})
    # Use the stable recurrence on the full wavefunctions (not raw Hermite polynomials)
    # φ_0(q) = π^{-1/4} exp(-q²/2)
    # φ_{n+1}(q) = √(2/(n+1)) q φ_n(q) - √(n/(n+1)) φ_{n-1}(q)
    gauss = exp.(-q .^ 2 ./ 2.0)
    φ_prev = (π ^ (-0.25)) .* gauss  # φ_0

    n == 0 && return φ_prev

    φ_curr = sqrt(2.0) .* q .* φ_prev  # φ_1

    for k in 1:(n-1)
        φ_next = sqrt(2.0 / (k + 1)) .* q .* φ_curr .- sqrt(k / (k + 1)) .* φ_prev
        φ_prev = φ_curr
        φ_curr = φ_next
    end
    return φ_curr
end

"""
    _ho_basis_matrix(q, N_basis) -> Matrix{Float64}

Build the N_basis × length(q) matrix of harmonic oscillator wavefunctions
using a single-pass three-term recurrence.  Row n+1 holds φ_n(q).
"""
function _ho_basis_matrix(q::AbstractVector{<:Real}, N_basis::Int)
    Nq = length(q)
    Φ = Matrix{Float64}(undef, N_basis, Nq)

    gauss = exp.(-q .^ 2 ./ 2.0)
    Φ[1, :] .= (π ^ (-0.25)) .* gauss  # φ_0

    if N_basis >= 2
        Φ[2, :] .= sqrt(2.0) .* q .* Φ[1, :]  # φ_1
    end

    for n in 2:(N_basis - 1)
        @views Φ[n+1, :] .= sqrt(2.0 / n) .* q .* Φ[n, :] .- sqrt((n - 1) / n) .* Φ[n-1, :]
    end

    return Φ
end

"""
    _simpson_weights(q) -> Vector{Float64}

Composite Simpson's rule weights for non-uniform spacing.
Falls back to trapezoidal for odd-length input.
"""
function _simpson_weights(q::AbstractVector{<:Real})
    N = length(q)
    w = zeros(Float64, N)

    if N < 3 || isodd(N)
        # Trapezoidal fallback
        dq = diff(q)
        w[1] = dq[1] / 2
        for i in 2:(N-1)
            w[i] = (dq[i-1] + dq[i]) / 2
        end
        w[N] = dq[end] / 2
        return w
    end

    # Composite Simpson on pairs of intervals
    for i in 1:2:(N-2)
        h = q[i+2] - q[i]
        h1 = q[i+1] - q[i]
        h2 = q[i+2] - q[i+1]
        w[i]   += h * (2*h1 - h2) / (6*h1)
        w[i+1] += h^3 / (6*h1*h2)
        w[i+2] += h * (2*h2 - h1) / (6*h2)
    end

    return w
end

"""
    wavefunction_to_quantumobject(q, ψ; N_basis=25)

Convert a position-space wavefunction ψ(q) to a QuantumToolbox ket in the Fock basis.

Computes all N_basis overlaps ⟨n|ψ⟩ = ∫ φ_n(q) ψ(q) dq in a single pass
using the HO three-term recurrence and Simpson's rule integration.
"""
function wavefunction_to_quantumobject(q::AbstractVector{<:Real},
                                        ψ::AbstractVector{<:Number};
                                        N_basis::Int=25)
    w = _simpson_weights(q)

    # Normalize input: ∫|ψ|² dq = 1
    norm2 = dot(w, abs2.(ψ))
    ψ_norm = ψ ./ sqrt(norm2)

    # Weighted wavefunction: w .* ψ_norm
    wψ = w .* ψ_norm

    # Build basis matrix (N_basis × Nq) and compute all overlaps at once
    Φ = _ho_basis_matrix(q, N_basis)
    coeffs = Φ * wψ  # matrix-vector product: (N_basis × Nq) × (Nq,) → (N_basis,)

    state = QuantumObject(ComplexF64.(coeffs), dims=(N_basis,))
    return normalize(state)
end

# ── Constants ─────────────────────────────────────────────────────────────

export STATE_TYPES, LOSS_CONFIGS

const STATE_TYPES  = [:sqz_vac, :sqz_coh, :coherent_sup]
const LOSS_CONFIGS = [:loss_in, :loss_ch, :loss_out]

# ══════════════════════════════════════════════════════════════════════════
# Fock sparse superpositions:  |ψ⟩ = N Σ_k c_k |n_k⟩
#   3 reals per term: [n_float, c_re, c_im]
# ══════════════════════════════════════════════════════════════════════════

export fock_params_per_state, fock_unpack_params, fock_pack_params, fock_get_bounds,
       fock_create_state, fock_unpack_coeff_params, fock_pack_coeff_params, fock_get_coeff_bounds

fock_params_per_state() = 3

function fock_unpack_params(x::AbstractVector, n_sup::Int)
    @assert length(x) == 3 * n_sup "Expected $(3*n_sup) params, got $(length(x))"
    n_floats = Vector{Float64}(undef, n_sup)
    c_vals   = Vector{ComplexF64}(undef, n_sup)
    for k in 1:n_sup
        base       = (k - 1) * 3
        n_floats[k] = x[base + 1]
        c_vals[k]   = Complex(x[base + 2], x[base + 3])
    end
    return n_floats, c_vals
end

function fock_pack_params(n_floats::AbstractVector, c_vals::AbstractVector)::Vector{Float64}
    n_sup = length(n_floats)
    x = Vector{Float64}(undef, 3 * n_sup)
    for k in 1:n_sup
        base       = (k - 1) * 3
        x[base + 1] = n_floats[k]
        x[base + 2] = real(c_vals[k])
        x[base + 3] = imag(c_vals[k])
    end
    return x
end

function fock_get_bounds(n_sup::Int, N_basis::Int)
    lb = Float64[]
    ub = Float64[]
    for _ in 1:n_sup
        append!(lb, [0.0,  -1.0, -1.0])
        append!(ub, [Float64(N_basis - 1),  1.0,  1.0])
    end
    return lb, ub
end

function fock_unpack_coeff_params(x_coeff::AbstractVector, n_sup::Int)
    @assert length(x_coeff) == 2 * n_sup
    c_vals = Vector{ComplexF64}(undef, n_sup)
    for k in 1:n_sup
        base     = (k - 1) * 2
        c_vals[k] = Complex(x_coeff[base + 1], x_coeff[base + 2])
    end
    return c_vals
end

function fock_pack_coeff_params(c_vals::AbstractVector)::Vector{Float64}
    n_sup = length(c_vals)
    x = Vector{Float64}(undef, 2 * n_sup)
    for k in 1:n_sup
        base     = (k - 1) * 2
        x[base + 1] = real(c_vals[k])
        x[base + 2] = imag(c_vals[k])
    end
    return x
end

function fock_get_coeff_bounds(n_sup::Int)
    lb = fill(-1.0, 2 * n_sup)
    ub = fill( 1.0, 2 * n_sup)
    return lb, ub
end

"""
    fock_create_state(n_floats, c_vals, N_basis) -> ket

Builds a normalised Fock superposition |ψ⟩ = N Σ_k c_k |round(n_k)⟩.
"""
function fock_create_state(n_floats::AbstractVector, c_vals::AbstractVector, N_basis::Int)
    ψ_data = zeros(ComplexF64, N_basis)
    for k in eachindex(c_vals)
        abs(c_vals[k]) < 1e-12 && continue
        n_k = clamp(round(Int, n_floats[k]), 0, N_basis - 1)
        ψ_data[n_k + 1] += c_vals[k]
    end
    if norm(ψ_data) < 1e-10
        ψ_data[1] = 1.0
    end
    return normalize(QuantumObject(ψ_data, dims=(N_basis,)))
end

# ══════════════════════════════════════════════════════════════════════════
# Squeezed/coherent superpositions:  |ψ⟩ = N Σ_k c_k D(α_k)S(r_k)|0⟩
#   :sqz_vac      — 4 reals: [c_re, c_im, r_re, r_im]
#   :sqz_coh      — 6 reals: [c_re, c_im, α_re, α_im, r_re, r_im]
#   :coherent_sup — 4 reals: [c_re, c_im, α_re, α_im]
# ══════════════════════════════════════════════════════════════════════════

export sqz_params_per_state, sqz_unpack_params, sqz_pack_params, sqz_get_bounds,
       sqz_create_state, embed_params_into_sqz_coh

function sqz_params_per_state(state_type::Symbol)::Int
    state_type == :sqz_vac     && return 4
    state_type == :sqz_coh     && return 6
    state_type == :coherent_sup && return 4
    error("Unknown state_type: $state_type. Choose from $STATE_TYPES")
end

function sqz_unpack_params(state_type::Symbol, x::AbstractVector, n_sups::Int)
    p = sqz_params_per_state(state_type)
    @assert length(x) == p * n_sups "Expected $(p*n_sups) params, got $(length(x))"

    c_vals     = Vector{ComplexF64}(undef, n_sups)
    alpha_vals = zeros(ComplexF64, n_sups)
    r_vals     = zeros(ComplexF64, n_sups)

    for k in 1:n_sups
        base = (k - 1) * p
        if state_type == :sqz_vac
            c_vals[k] = Complex(x[base+1], x[base+2])
            r_vals[k] = Complex(x[base+3], x[base+4])
        elseif state_type == :sqz_coh
            c_vals[k]     = Complex(x[base+1], x[base+2])
            alpha_vals[k] = Complex(x[base+3], x[base+4])
            r_vals[k]     = Complex(x[base+5], x[base+6])
        elseif state_type == :coherent_sup
            c_vals[k]     = Complex(x[base+1], x[base+2])
            alpha_vals[k] = Complex(x[base+3], x[base+4])
        end
    end
    return c_vals, alpha_vals, r_vals
end

function sqz_pack_params(state_type::Symbol,
                         c_vals::AbstractVector,
                         alpha_vals::AbstractVector,
                         r_vals::AbstractVector)::Vector{Float64}
    n_sups = length(c_vals)
    p = sqz_params_per_state(state_type)
    x = Vector{Float64}(undef, p * n_sups)

    for k in 1:n_sups
        base = (k - 1) * p
        if state_type == :sqz_vac
            x[base+1] = real(c_vals[k]);     x[base+2] = imag(c_vals[k])
            x[base+3] = real(r_vals[k]);     x[base+4] = imag(r_vals[k])
        elseif state_type == :sqz_coh
            x[base+1] = real(c_vals[k]);     x[base+2] = imag(c_vals[k])
            x[base+3] = real(alpha_vals[k]); x[base+4] = imag(alpha_vals[k])
            x[base+5] = real(r_vals[k]);     x[base+6] = imag(r_vals[k])
        elseif state_type == :coherent_sup
            x[base+1] = real(c_vals[k]);     x[base+2] = imag(c_vals[k])
            x[base+3] = real(alpha_vals[k]); x[base+4] = imag(alpha_vals[k])
        end
    end
    return x
end

function sqz_get_bounds(state_type::Symbol, n_sups::Int, N_target::Real)
    c_lim     = 1.0
    alpha_lim = sqrt(max(N_target, 1.0))
    r_lim     = max(asinh(2 * N_target), 2.0)

    lb = Float64[]
    ub = Float64[]

    for _ in 1:n_sups
        if state_type == :sqz_vac
            append!(lb, [-c_lim, -c_lim, -r_lim, -r_lim])
            append!(ub, [ c_lim,  c_lim,  r_lim,  r_lim])
        elseif state_type == :sqz_coh
            append!(lb, [-c_lim, -c_lim, -alpha_lim, -alpha_lim, -r_lim, -r_lim])
            append!(ub, [ c_lim,  c_lim,  alpha_lim,  alpha_lim,  r_lim,  r_lim])
        elseif state_type == :coherent_sup
            append!(lb, [-c_lim, -c_lim, -alpha_lim, -alpha_lim])
            append!(ub, [ c_lim,  c_lim,  alpha_lim,  alpha_lim])
        end
    end
    return lb, ub
end

function _create_component(α::Number, r::Number, N_basis::Int)
    ψ = fock(N_basis, 0)
    abs(r) > 1e-10 && (ψ = squeeze(N_basis, r) * ψ)
    abs(α) > 1e-10 && (ψ = displace(N_basis, α) * ψ)
    return ψ
end

"""
    sqz_create_state(c_vals, alpha_vals, r_vals, N_basis) -> ket

Builds a normalised superposition  |ψ⟩ = N Σ_k c_k D(α_k)S(r_k)|0⟩.
"""
function sqz_create_state(c_vals::AbstractVector,
                          alpha_vals::AbstractVector,
                          r_vals::AbstractVector,
                          N_basis::Int)
    n = length(c_vals)
    ψ_super = zeros(ComplexF64, N_basis)

    for k in 1:n
        abs(c_vals[k]) < 1e-12 && continue
        ψ_k = _create_component(alpha_vals[k], r_vals[k], N_basis)
        ψ_super .+= c_vals[k] .* ψ_k.data
    end

    return normalize(QuantumObject(ψ_super, dims=(N_basis,)))
end

"""
    embed_params_into_sqz_coh(source_type, source_params, n_sups) -> Vector{Float64}

Converts a parameter vector from :sqz_vac or :coherent_sup into the :sqz_coh layout.
"""
function embed_params_into_sqz_coh(source_type::Symbol,
                                    source_params::AbstractVector,
                                    n_sups::Int)::Vector{Float64}
    p_src = sqz_params_per_state(source_type)
    @assert length(source_params) == p_src * n_sups

    x_cat = zeros(Float64, 6 * n_sups)

    for k in 1:n_sups
        src_base  = (k - 1) * p_src
        cat_base  = (k - 1) * 6

        c_re, c_im = source_params[src_base+1], source_params[src_base+2]
        x_cat[cat_base+1] = c_re
        x_cat[cat_base+2] = c_im

        if source_type == :sqz_vac
            x_cat[cat_base+5] = source_params[src_base+3]
            x_cat[cat_base+6] = source_params[src_base+4]
        elseif source_type == :coherent_sup
            x_cat[cat_base+3] = source_params[src_base+3]
            x_cat[cat_base+4] = source_params[src_base+4]
        end
    end

    return x_cat
end
