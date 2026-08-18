# GPU dynamics — pre-built operator constructors and GPU-accelerated evolution.

export build_mzi_operators, build_single_mode_operators

"""
    build_single_mode_operators(N)

Construct GPU operators for `get_state_single_mode`.
"""
function QuantumSensing.build_single_mode_operators(N::Int)
    a      = cu(destroy(N))
    n      = a' * a
    H_zero = 0 * n
    return SingleModeOperators(N, a, n, H_zero)
end

"""
    build_mzi_operators(N)

Construct GPU operators for `get_state_mzi_heis` and `get_state_arms`.
"""
function QuantumSensing.build_mzi_operators(N::Int)
    a_cpu  = destroy(N)
    id_cpu = qeye(N)

    aC = cu(tensor(a_cpu, id_cpu))
    aD = cu(tensor(id_cpu, a_cpu))
    NC = aC' * aC
    ND = aD' * aD

    inv_sqrt2 = 1.0 / √2
    aA = (aC + aD) * inv_sqrt2
    aB = (aC - aD) * inv_sqrt2
    NA = aA' * aA
    NB = aB' * aB

    H_zero_two = 0 * NC

    return MZIOperators(N, aC, aD, NC, ND, aA, aB, NA, NB, H_zero_two)
end

# Move ρ to GPU only if not already there
function _ensure_gpu(ρ::QuantumObject)
    d = ρ.data
    (d isa CuArray || d isa CUDA.CUSPARSE.CuSparseMatrixCSC) ? ρ : cu(ρ)
end

# Multi-channel noise stage
function _apply_noise_mzi(ρ, t, η_vals, pn_vals, a_ops, n_ops, H_zero)
    all(η -> η >= 1.0, η_vals) && all(p -> p <= 0.0, pn_vals) && return ρ

    L = QuantumObject[]
    for (i, η) in enumerate(η_vals)
        η < 1.0 && push!(L, sqrt(loss_to_kappa(1.0 - η, t)) * a_ops[i])
    end
    for (i, pn) in enumerate(pn_vals)
        pn > 0.0 && push!(L, sqrt(phirms_to_chi(pn, t)) * n_ops[i])
    end

    isempty(L) && return ρ
    return mesolve(H_zero, ρ, [0.0, t], L; progress_bar=Val(false)).states[end]
end

# ── GPU get_state_mzi_heis ────────────────────────────────────────────────

function QuantumSensing.get_state_mzi_heis(
    ops::MZIOperators;
    Δ::Real=0.0,
    ϵ_a::Real=0.0,
    ϵ_p::Real=0.0,
    η_arms::Tuple{Real,Real}=(1.0, 1.0),
    pn_arms::Tuple{Real,Real}=(0.0, 0.0),
    η_in::Tuple{Real,Real}=(1.0, 1.0),
    pn_in::Tuple{Real,Real}=(0.0, 0.0),
    η_out::Tuple{Real,Real}=(1.0, 1.0),
    pn_out::Tuple{Real,Real}=(0.0, 0.0),
    t_final::Real=1.0,
    ρ=nothing,
    sensing_mode::Symbol=:single,
    show_progress=false,
)
    if isnothing(ρ)
        ρ_curr = cu(ket2dm(tensor(fock(ops.N_basis, 0), fock(ops.N_basis, 0))))
    else
        ρ_curr = isket(ρ) ? ket2dm(ρ) : ρ
        ρ_curr = _ensure_gpu(ρ_curr)
    end

    # Stage 1 — input noise
    ρ_curr = _apply_noise_mzi(ρ_curr, 1.0, η_in, pn_in,
        [ops.aC, ops.aD], [ops.NC, ops.ND], ops.H_zero_two)

    # Stage 2 — arm dynamics
    if sensing_mode == :single
        H = Δ * ops.NA +
            ϵ_a * (ops.aA' + ops.aA) +
            1im * ϵ_p * (ops.aA' - ops.aA)
    elseif sensing_mode == :differential
        H = Δ * (ops.NA - ops.NB) +
            ϵ_a * ((ops.aA' + ops.aA) - (ops.aB' + ops.aB)) +
            1im * ϵ_p * ((ops.aA' - ops.aA) - (ops.aB' - ops.aB))
    else
        error("sensing_mode must be :single or :differential")
    end

    L_arms = QuantumObject[]
    η_arms[1] < 1.0  && push!(L_arms, sqrt(loss_to_kappa(1.0 - η_arms[1], t_final)) * ops.aA)
    η_arms[2] < 1.0  && push!(L_arms, sqrt(loss_to_kappa(1.0 - η_arms[2], t_final)) * ops.aB)
    pn_arms[1] > 0.0 && push!(L_arms, sqrt(phirms_to_chi(pn_arms[1], t_final)) * ops.NA)
    pn_arms[2] > 0.0 && push!(L_arms, sqrt(phirms_to_chi(pn_arms[2], t_final)) * ops.NB)

    res_arms = mesolve(H, ρ_curr, [0.0, t_final], L_arms; progress_bar=Val(show_progress))
    ρ_curr = res_arms.states[end]

    # Stage 3 — output noise
    ρ_curr = _apply_noise_mzi(ρ_curr, 1.0, η_out, pn_out,
        [ops.aC, ops.aD], [ops.NC, ops.ND], ops.H_zero_two)

    return ρ_curr
end

# ── GPU get_state_arms ────────────────────────────────────────────────────

function QuantumSensing.get_state_arms(
    ops::MZIOperators;
    Δ::Real=0.0,
    ηx::Real=1.0,
    ηy::Real=1.0,
    pnx::Real=0.0,
    pny::Real=0.0,
    t_final::Real=1.0,
    ρ=nothing,
)
    if isnothing(ρ)
        ρ_curr = cu(tensor(ket2dm(coherent(ops.N_basis, 1.0)),
                           ket2dm(fock(ops.N_basis, 0))))
    else
        ρ_curr = isket(ρ) ? ket2dm(ρ) : ρ
        ρ_curr = _ensure_gpu(ρ_curr)
    end

    if Δ != 0.0
        ρ_curr = mesolve(Δ * ops.NC, ρ_curr, [0.0, t_final], QuantumObject[];
                         progress_bar=Val(false)).states[end]
    end

    c_ops = QuantumObject[]
    ηx < 1.0  && push!(c_ops, sqrt(loss_to_kappa(1.0 - ηx)) * ops.aC)
    ηy < 1.0  && push!(c_ops, sqrt(loss_to_kappa(1.0 - ηy)) * ops.aD)
    pnx > 0.0 && push!(c_ops, sqrt(phirms_to_chi(pnx)) * ops.NC)
    pny > 0.0 && push!(c_ops, sqrt(phirms_to_chi(pny)) * ops.ND)

    isempty(c_ops) && return ρ_curr
    return mesolve(ops.H_zero_two, ρ_curr, [0.0, t_final], c_ops;
                   progress_bar=Val(false)).states[end]
end

# ── GPU get_state_single_mode ─────────────────────────────────────────────

function QuantumSensing.get_state_single_mode(
    ops::SingleModeOperators;
    Δ::Real=0.0,
    ϵ_a::Real=0.0,
    ϵ_p::Real=0.0,
    η_in::Real=1.0,  η_ch::Real=1.0,  η_out::Real=1.0,
    pn_in::Real=0.0, pn_ch::Real=0.0, pn_out::Real=0.0,
    σ_a::Real=0.0,
    σ_p::Real=0.0,
    t_final::Real=1.0,
    ρ=nothing,
)
    a      = ops.a
    n      = ops.n
    H_zero = ops.H_zero

    if isnothing(ρ)
        ρ_curr = cu(ket2dm(coherent(ops.N_basis, 1.0)))
    else
        ρ_curr = isket(ρ) ? ket2dm(ρ) : ρ
        ρ_curr = _ensure_gpu(ρ_curr)
    end

    # Input noise
    L_in = QuantumObject[]
    η_in  < 1.0 && push!(L_in, sqrt(loss_to_kappa(1.0 - η_in)) * a)
    pn_in > 0.0 && push!(L_in, sqrt(phirms_to_chi(pn_in)) * n)
    if !isempty(L_in)
        ρ_curr = mesolve(H_zero, ρ_curr, [0.0, 1.0], L_in;
                         progress_bar=Val(false)).states[end]
    end

    # Sensing channel
    H    = Δ * n + ϵ_a * (a' + a) + (1im * ϵ_p) * (a' - a)
    L_ch = QuantumObject[]
    η_ch  < 1.0 && push!(L_ch, sqrt(loss_to_kappa(1.0 - η_ch, t_final)) * a)
    pn_ch > 0.0 && push!(L_ch, sqrt(phirms_to_chi(pn_ch, t_final)) * n)
    σ_a   > 0.0 && push!(L_ch, sqrt(σ_a) * (a' + a))
    σ_p   > 0.0 && push!(L_ch, 1im * sqrt(σ_p) * (a' - a))
    if Δ != 0.0 || ϵ_a != 0.0 || ϵ_p != 0.0 || !isempty(L_ch)
        ρ_curr = mesolve(H, ρ_curr, [0.0, t_final], L_ch;
                         progress_bar=Val(false)).states[end]
    end

    # Output noise
    L_out = QuantumObject[]
    η_out  < 1.0 && push!(L_out, sqrt(loss_to_kappa(1.0 - η_out)) * a)
    pn_out > 0.0 && push!(L_out, sqrt(phirms_to_chi(pn_out)) * n)
    if !isempty(L_out)
        ρ_curr = mesolve(H_zero, ρ_curr, [0.0, 1.0], L_out;
                         progress_bar=Val(false)).states[end]
    end

    return ρ_curr
end

# ── GPU apply_bs ──────────────────────────────────────────────────────────

function QuantumSensing.apply_bs(ρ_in::QuantumObject, ops::MZIOperators;
                                  reverse::Bool=false, θ=π/4)
    G_sparse = ops.aC' * ops.aD - ops.aD' * ops.aC
    θ_val    = reverse ? -θ : θ
    U_data   = exp(θ_val * CuArray(G_sparse.data))
    U        = QuantumObject(U_data, type=Operator, dims=ρ_in.dims)
    return isket(ρ_in) ? U * ρ_in : U * ρ_in * U'
end
