# CPU dynamics — Lindblad evolution for MZI, two-arm, and single-mode sensing.

export get_state_mzi_heis, get_state_arms, get_state_single_mode, apply_bs

"""
Mach-Zehnder interferometer: Heisenberg picture formulation.

The density matrix remains fixed in the computational basis (C, D modes).
Beam splitters are encoded as operator transformations:
  a_A = (a_C + a_D)/√2  (arm A)
  a_B = (a_C - a_D)/√2  (arm B)

Evolution sequence:
1. Input noise: Lindblads √κ a_C, √κ a_D act on ρ
2. Arm dynamics: H and Lindblads expressed in arm operators {a_A, a_B}
3. Output noise: Lindblads √κ a_C, √κ a_D act on ρ
"""
function get_state_mzi_heis(;
    Δ::Real=0.0,
    ϵ_a::Real=0.0,
    ϵ_p::Real=0.0,
    η_arms::Tuple{Real, Real}=(1.0, 1.0),
    pn_arms::Tuple{Real, Real}=(0.0, 0.0),
    η_in::Tuple{Real, Real}=(1.0, 1.0),
    pn_in::Tuple{Real, Real}=(0.0, 0.0),
    η_out::Tuple{Real, Real}=(1.0, 1.0),
    pn_out::Tuple{Real, Real}=(0.0, 0.0),
    t_final::Real=1.0,
    ρ=nothing,
    N_basis::Int=20,
    sensing_mode::Symbol=:single,
    show_progress=false
)
    # Computational Basis Operators (C, D modes)
    aC = tensor(destroy(N_basis), qeye(N_basis))
    aD = tensor(qeye(N_basis), destroy(N_basis))
    NC = aC' * aC
    ND = aD' * aD

    # Arm Basis Operators
    aA = (aC + aD) / √2
    aB = (aC - aD) / √2
    NA = aA' * aA
    NB = aB' * aB

    # Input State
    if isnothing(ρ)
        ψ_vac = tensor(fock(N_basis, 0), fock(N_basis, 0))
        ρ = ket2dm(ψ_vac)
    else
        ρ = isket(ρ) ? ket2dm(ρ) : ρ
    end

    # Noise Application Helper
    function apply_noise_stage(ρ_curr, η_vals, pn_vals, ops, N_ops)
        L_stage = QuantumObject[]
        t_stage = 1.0

        for (i, η) in enumerate(η_vals)
            if η < 1.0
                κ = loss_to_kappa(1 - η, t_stage)
                push!(L_stage, √κ * ops[i])
            end
        end

        for (i, pn) in enumerate(pn_vals)
            if pn > 0.0
                χ = phirms_to_chi(pn, t_stage)
                push!(L_stage, √χ * N_ops[i])
            end
        end

        if isempty(L_stage)
            return ρ_curr
        else
            res = mesolve(0*ops[1], ρ_curr, [0, t_stage], L_stage,
                         progress_bar=Val(false))
            return res.states[end]
        end
    end

    # Stage 1: Input Noise (C, D operators)
    ρ = apply_noise_stage(ρ, η_in, pn_in, [aC, aD], [NC, ND])

    # Stage 2: Arm Dynamics
    if sensing_mode == :single
        H = Δ * NA + ϵ_a * (aA' + aA) + 1im * ϵ_p * (aA' - aA)
    elseif sensing_mode == :differential
        H = Δ * (NA - NB) +
            ϵ_a * ((aA' + aA) - (aB' + aB)) +
            1im * ϵ_p * ((aA' - aA) - (aB' - aB))
    else
        error("sensing_mode must be :single or :differential")
    end

    L_arms = QuantumObject[]
    if η_arms[1] < 1.0
        κA = loss_to_kappa(1 - η_arms[1], t_final)
        push!(L_arms, √κA * aA)
    end
    if η_arms[2] < 1.0
        κB = loss_to_kappa(1 - η_arms[2], t_final)
        push!(L_arms, √κB * aB)
    end
    if pn_arms[1] > 0.0
        χA = phirms_to_chi(pn_arms[1], t_final)
        push!(L_arms, √χA * NA)
    end
    if pn_arms[2] > 0.0
        χB = phirms_to_chi(pn_arms[2], t_final)
        push!(L_arms, √χB * NB)
    end

    res_arms = mesolve(H, ρ, range(0, t_final, length=2), L_arms,
                       progress_bar=Val(show_progress))
    ρ = res_arms.states[end]

    # Stage 3: Output Noise (C, D operators)
    ρ = apply_noise_stage(ρ, η_out, pn_out, [aC, aD], [NC, ND])

    return ρ
end


function get_state_arms(;
    Δ=0.0,
    ηx=1.0,
    ηy=1.0,
    pnx=0.0,
    pny=0.0,
    t_final=1.0,
    N_basis=20,
    ρx=nothing, ρy=nothing, ρ=nothing
)
    if isnothing(ρ)
        ρx0 = isnothing(ρx) ? coherent(N_basis, 1.0) : ρx
        ρy0 = isnothing(ρx) ? basis(N_basis, 0) : ρx
        ρ0 = tensor(ρx0, ρy0)
    else
        ρ0 = ρ
    end
    ρ0 = isket(ρ0) ? ket2dm(ρ0) : ρ0

    ax = tensor(destroy(N_basis), qeye(N_basis))
    ay = tensor(qeye(N_basis), destroy(N_basis))

    # Sensing
    H = Δ * ax' * ax
    t_list = [0.0, t_final]
    result = mesolve(H, ρ0, t_list, progress_bar=false)
    ρ_sensed = result.states[end]

    # Loss and dephasing
    c_ops = QuantumObject[]

    if ηx < 1.0 || ηy < 1.0
        κx = loss_to_kappa(1 - ηx)
        κy = loss_to_kappa(1 - ηy)
        push!(c_ops, sqrt(κx) * ax)
        push!(c_ops, sqrt(κy) * ay)
    end

    if pnx > 0.0 || pny > 0.0
        χx = phirms_to_chi(pnx)
        χy = phirms_to_chi(pny)
        push!(c_ops, sqrt(χx) * ax' * ax)
        push!(c_ops, sqrt(χy) * ay' * ay)
    end

    if !isempty(c_ops)
        H1 = 0.0 * ax' * ax
        result = mesolve(H1, ρ_sensed, t_list, c_ops, progress_bar=false)
        return result.states[end]
    end

    ρ_sensed
end

"""
Single mode state evolution with 3-stage noise pipeline:
  1. Input noise (η_in, pn_in)
  2. Sensing channel (H + channel loss/dephasing/amplitude/phase noise)
  3. Output noise (η_out, pn_out)
"""
function get_state_single_mode(;
    Δ::Real=0.0,
    ϵ_a::Real=0.0,
    ϵ_p::Real=0.0,
    η_in::Real=1.0,
    η_ch::Real=1.0,
    η_out::Real=1.0,
    pn_in::Real=0.0,
    pn_ch::Real=0.0,
    pn_out::Real=0.0,
    σ_a::Real=0.0,
    σ_p::Real=0.0,
    t_final::Real=1.0,
    N_basis::Int=20,
    ρ=nothing
)
    if isnothing(ρ)
        ρ0 = coherent(N_basis, 1.0)
    else
        ρ0 = isket(ρ) ? ket2dm(ρ) : ρ
    end

    a = destroy(N_basis)
    n = a' * a

    # Input noise
    L_in = QuantumObject[]
    t_in = 1.0
    if η_in < 1.0
        κ_in = loss_to_kappa(1 - η_in)
        push!(L_in, sqrt(κ_in) * a)
    end
    if pn_in > 0.0
        χ_in = phirms_to_chi(pn_in)
        push!(L_in, sqrt(χ_in) * n)
    end

    if isempty(L_in)
        ρ1 = ρ0
    else
        res_in = mesolve(0*n, ρ0, [0, t_in], L_in, progress_bar=Val(false))
        ρ1 = res_in.states[end]
    end

    # Sensing Channel
    H = Δ * n + ϵ_a * (a' + a) + 1im * ϵ_p * (a' - a)
    t_list = [0.0, t_final]

    L_ops = QuantumObject[]
    if η_ch < 1.0
        κ = loss_to_kappa(1 - η_ch)
        push!(L_ops, sqrt(κ) * a)
    end
    if pn_ch > 0.0
        χ = phirms_to_chi(pn_ch)
        push!(L_ops, sqrt(χ) * n)
    end
    if σ_a > 0.0
        push!(L_ops, sqrt(σ_a) * (a' + a))
    end
    if σ_p > 0.0
        push!(L_ops, 1im * sqrt(σ_p) * (a' - a))
    end

    res_ch = mesolve(H, ρ1, t_list, L_ops, progress_bar=false)
    ρ_sensed = res_ch.states[end]

    # Output noise
    L_out = QuantumObject[]
    t_out = 1.0
    if η_out < 1.0
        κ_out = loss_to_kappa(1 - η_out)
        push!(L_out, sqrt(κ_out) * a)
    end
    if pn_out > 0.0
        χ_out = phirms_to_chi(pn_out)
        push!(L_out, sqrt(χ_out) * n)
    end

    if isempty(L_out)
        ρout = ρ_sensed
    else
        res_out = mesolve(0*n, ρ_sensed, [0, t_out], L_out, progress_bar=Val(false))
        ρout = res_out.states[end]
    end

    return ρout
end

function apply_bs(ρ_in::QuantumObject, N_basis::Int; reverse::Bool=false, θ=π/4)
    aC = tensor(destroy(N_basis), qeye(N_basis))
    aD = tensor(qeye(N_basis), destroy(N_basis))

    θ = reverse ? -θ : θ
    G_BS = aC' * aD - aD' * aC
    U_BS = exp(θ * G_BS)

    if isket(ρ_in)
        return U_BS * ρ_in
    else
        ρ_arms = U_BS * ρ_in * U_BS'
        return ρ_arms
    end
end

# CPU fallback: accept ops=nothing (GPU ops dispatch goes through the CUDA extension)
get_state_mzi_heis(::Nothing; kwargs...) = get_state_mzi_heis(; kwargs...)
get_state_arms(::Nothing; kwargs...) = get_state_arms(; kwargs...)
get_state_single_mode(::Nothing; kwargs...) = get_state_single_mode(; kwargs...)
