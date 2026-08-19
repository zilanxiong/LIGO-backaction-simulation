# Classical Fisher Information for various measurement types.

export calculate_cfi, calculate_cfi_photon_counting, calculate_cfi_homodyne,
       calculate_cfi_heterodyne, calculate_cfi_sld_optimal, calculate_cfi_projective,
       compare_cfi

const PREC_DIFF_CFI = 1e-5
const PROB_FLOOR = 1e-15

# ── Internal helpers ──────────────────────────────────────────────────────

function _get_rho_drho(; dynamics::Function, param_value::Real=0.0,
                        param_type::Symbol=:Δ, prec::Real=PREC_DIFF_CFI, kwargs...)
    params_plus  = merge(kwargs, Dict(param_type => param_value + prec))
    params_minus = merge(kwargs, Dict(param_type => param_value - prec))
    params_center = merge(kwargs, Dict(param_type => param_value))

    function get_dense_dm(params)
        state = dynamics(; params...)
        if isket(state)
            state = ket2dm(state)
        end
        return Matrix{ComplexF64}(state.data)
    end

    ρ_plus  = get_dense_dm(params_plus)
    ρ_minus = get_dense_dm(params_minus)
    ρ_mat   = get_dense_dm(params_center)
    dρ_mat  = (ρ_plus - ρ_minus) / (2 * prec)

    return ρ_mat, dρ_mat
end

function _cfi_from_probs(probs::AbstractVector{<:Real},
                         dprobs::AbstractVector{<:Real};
                         prob_floor::Real=PROB_FLOOR)
    cfi = 0.0
    @inbounds for i in eachindex(probs)
        p = probs[i]
        if p > prob_floor
            cfi += dprobs[i]^2 / p
        end
    end
    return cfi
end

function _coherent_state_vec!(v::Vector{ComplexF64}, alpha::ComplexF64)
    N = length(v)
    v[1] = exp(-abs2(alpha) / 2)
    for n in 1:(N-1)
        v[n+1] = v[n] * alpha / sqrt(n)
    end
    return v
end

# ── Measurement-specific CFI functions ────────────────────────────────────

function calculate_cfi_photon_counting(rho::AbstractMatrix{<:Number},
                                       drho::AbstractMatrix{<:Number})
    N = size(rho, 1)
    probs  = Float64[real(rho[m, m]) for m in 1:N]
    dprobs = Float64[real(drho[m, m]) for m in 1:N]
    cfi = _cfi_from_probs(probs, dprobs)
    return (cfi=cfi, probabilities=probs, dprobabilities=dprobs,
            measurement_type=:photon_counting)
end

function calculate_cfi_homodyne(rho::AbstractMatrix{<:Number},
                                drho::AbstractMatrix{<:Number};
                                theta_LO::Real=0.0)
    N_basis = size(rho, 1)
    a = destroy(N_basis)
    X_theta = (a * exp(-im * theta_LO) + a' * exp(im * theta_LO)) / sqrt(2)
    evals, evecs = eigen(Hermitian(Matrix(X_theta.data)))

    probs  = Float64[real(dot(evecs[:, m], rho  * evecs[:, m])) for m in 1:N_basis]
    dprobs = Float64[real(dot(evecs[:, m], drho * evecs[:, m])) for m in 1:N_basis]
    probs .= max.(probs, 0.0)

    cfi = _cfi_from_probs(probs, dprobs)
    return (cfi=cfi, probabilities=probs, dprobabilities=dprobs,
            theta_LO=theta_LO, measurement_type=:homodyne)
end

function calculate_cfi_sld_optimal(rho::AbstractMatrix{<:Number},
                                    drho::AbstractMatrix{<:Number};
                                    sld_eigenvectors::Union{Nothing,AbstractMatrix}=nothing,
                                    prec_eig::Real=1e-8)
    N = size(rho, 1)

    if sld_eigenvectors === nothing
        evals_rho, evecs_rho = eigen(Hermitian(Matrix(rho)))
        dρ_rot = evecs_rho' * drho * evecs_rho
        L_rot = zeros(ComplexF64, N, N)
        for i in 1:N, j in 1:N
            denom = evals_rho[i] + evals_rho[j]
            if abs(denom) > prec_eig
                L_rot[i, j] = 2.0 * dρ_rot[i, j] / denom
            end
        end
        L_mat = evecs_rho * L_rot * evecs_rho'
        L_mat = (L_mat + L_mat') / 2.0
        qfi = real(tr(rho * L_mat^2))
        sld_evals, sld_evecs = eigen(Hermitian(L_mat))
    else
        sld_evecs = sld_eigenvectors
        sld_evals = Float64[real(dot(sld_evecs[:, m], drho * sld_evecs[:, m])) /
                            max(real(dot(sld_evecs[:, m], rho * sld_evecs[:, m])), PROB_FLOOR)
                            for m in 1:N]
        evals_rho, evecs_rho = eigen(Hermitian(Matrix(rho)))
        dρ_rot = evecs_rho' * drho * evecs_rho
        qfi = 0.0
        for i in 1:N, j in 1:N
            denom = evals_rho[i] + evals_rho[j]
            if abs(denom) > prec_eig
                qfi += 2.0 * abs2(dρ_rot[i, j]) / denom
            end
        end
        qfi = real(qfi)
    end

    probs  = Float64[real(dot(sld_evecs[:, m], rho  * sld_evecs[:, m])) for m in 1:N]
    dprobs = Float64[real(dot(sld_evecs[:, m], drho * sld_evecs[:, m])) for m in 1:N]
    cfi = _cfi_from_probs(probs, dprobs)

    return (cfi=cfi, probabilities=probs, dprobabilities=dprobs,
            sld_eigenvalues=sld_evals, sld_eigenvectors=sld_evecs,
            qfi=qfi, measurement_type=:sld_optimal)
end

function calculate_cfi_projective(rho::AbstractMatrix{<:Number},
                                   drho::AbstractMatrix{<:Number},
                                   projectors::Vector{<:AbstractVector{<:Number}})
    M = length(projectors)
    probs  = Float64[real(dot(projectors[m], rho  * projectors[m])) for m in 1:M]
    dprobs = Float64[real(dot(projectors[m], drho * projectors[m])) for m in 1:M]
    cfi = _cfi_from_probs(probs, dprobs)
    return (cfi=cfi, probabilities=probs, dprobabilities=dprobs,
            measurement_type=:projective)
end

function calculate_cfi_heterodyne(rho::AbstractMatrix{<:Number},
                                   drho::AbstractMatrix{<:Number};
                                   n_grid::Int=60,
                                   alpha_max::Union{Real,Nothing}=nothing,
                                   N_target::Union{Real,Nothing}=nothing)
    N_basis = size(rho, 1)

    if alpha_max === nothing
        if N_target !== nothing
            alpha_max = max(3.0, 1.5 * sqrt(N_target + 1))
        else
            a = destroy(N_basis)
            n_op = Matrix(a' * a)
            n_bar = real(tr(rho * n_op))
            alpha_max = max(3.0, 1.5 * sqrt(n_bar + 3 * sqrt(max(n_bar, 0.0)) + 1))
        end
    end

    alpha_re = range(-alpha_max, alpha_max; length=n_grid)
    alpha_im = range(-alpha_max, alpha_max; length=n_grid)
    dA = step(alpha_re) * step(alpha_im)

    n_total = n_grid * n_grid
    raw_probs  = Vector{Float64}(undef, n_total)
    raw_dprobs = Vector{Float64}(undef, n_total)
    v   = Vector{ComplexF64}(undef, N_basis)
    tmp = Vector{ComplexF64}(undef, N_basis)

    k = 0
    for ai in alpha_im, ar in alpha_re
        k += 1
        α = ComplexF64(ar, ai)
        _coherent_state_vec!(v, α)
        mul!(tmp, rho, v)
        raw_probs[k] = max(real(dot(v, tmp)), 0.0)
        mul!(tmp, drho, v)
        raw_dprobs[k] = real(dot(v, tmp))
    end

    cfi = _cfi_from_probs(raw_probs, raw_dprobs) * dA

    normalization = sum(raw_probs) * dA / π
    if abs(normalization - 1.0) > 0.05
        @warn "Heterodyne grid normalization = $(round(normalization; digits=4)) " *
              "(expected ≈ 1.0). Consider increasing n_grid or alpha_max."
    end

    return (cfi=cfi, alpha_max=alpha_max, n_grid=n_grid,
            normalization=normalization, measurement_type=:heterodyne)
end

# ── Unified entry point ──────────────────────────────────────────────────

function calculate_cfi(; dynamics::Union{Function,Nothing}=nothing,
                        param_value::Real=0.0,
                        param_type::Symbol=:Δ,
                        prec::Real=PREC_DIFF_CFI,
                        rho::Union{Nothing,AbstractMatrix}=nothing,
                        drho::Union{Nothing,AbstractMatrix}=nothing,
                        measurement::Symbol=:photon_counting,
                        kwargs...)
    if rho === nothing || drho === nothing
        dynamics === nothing && error("Must provide either `dynamics` or both `rho` and `drho`")
        rho, drho = _get_rho_drho(; dynamics=dynamics, param_value=param_value,
                                    param_type=param_type, prec=prec, kwargs...)
    end

    if measurement == :photon_counting
        return calculate_cfi_photon_counting(rho, drho)
    elseif measurement == :homodyne
        theta_LO = get(kwargs, :theta_LO, 0.0)
        return calculate_cfi_homodyne(rho, drho; theta_LO=theta_LO)
    elseif measurement == :heterodyne
        n_grid = get(kwargs, :n_grid, 60)
        alpha_max = get(kwargs, :alpha_max, nothing)
        N_target = get(kwargs, :N_target, nothing)
        return calculate_cfi_heterodyne(rho, drho; n_grid=n_grid,
                                        alpha_max=alpha_max, N_target=N_target)
    elseif measurement == :sld_optimal
        sld_eigenvectors = get(kwargs, :sld_eigenvectors, nothing)
        return calculate_cfi_sld_optimal(rho, drho; sld_eigenvectors=sld_eigenvectors)
    elseif measurement == :projective
        projectors = kwargs[:projectors]
        return calculate_cfi_projective(rho, drho, projectors)
    else
        error("Unknown measurement type: $measurement. " *
              "Use :photon_counting, :homodyne, :heterodyne, :sld_optimal, or :projective")
    end
end

# ── Compare multiple measurements ────────────────────────────────────────

function compare_cfi(; dynamics::Union{Function,Nothing}=nothing,
                      rho::Union{Nothing,AbstractMatrix}=nothing,
                      drho::Union{Nothing,AbstractMatrix}=nothing,
                      param_value::Real=0.0,
                      param_type::Symbol=:Δ,
                      prec::Real=PREC_DIFF_CFI,
                      measurements::Vector{Symbol}=[:photon_counting, :homodyne, :sld_optimal],
                      homodyne_phases::Vector{<:Real}=[0.0],
                      kwargs...)
    if rho === nothing || drho === nothing
        dynamics === nothing && error("Must provide either `dynamics` or both `rho` and `drho`")
        rho, drho = _get_rho_drho(; dynamics=dynamics, param_value=param_value,
                                    param_type=param_type, prec=prec, kwargs...)
    end

    results = Dict{Symbol, Any}()
    sld_result = calculate_cfi_sld_optimal(rho, drho)
    qfi = sld_result.qfi

    for meas in measurements
        if meas == :photon_counting
            results[:photon_counting] = calculate_cfi_photon_counting(rho, drho)
        elseif meas == :homodyne
            best_hom = nothing
            for θ_LO in homodyne_phases
                res = calculate_cfi_homodyne(rho, drho; theta_LO=θ_LO)
                if best_hom === nothing || res.cfi > best_hom.cfi
                    best_hom = res
                end
            end
            results[:homodyne] = best_hom
        elseif meas == :heterodyne
            het_kwargs = Dict{Symbol,Any}()
            haskey(kwargs, :n_grid) && (het_kwargs[:n_grid] = kwargs[:n_grid])
            haskey(kwargs, :alpha_max) && (het_kwargs[:alpha_max] = kwargs[:alpha_max])
            haskey(kwargs, :N_target) && (het_kwargs[:N_target] = kwargs[:N_target])
            results[:heterodyne] = calculate_cfi_heterodyne(rho, drho; het_kwargs...)
        elseif meas == :sld_optimal
            results[:sld_optimal] = sld_result
        elseif meas == :projective
            if haskey(kwargs, :projectors)
                results[:projective] = calculate_cfi_projective(rho, drho, kwargs[:projectors])
            end
        end
    end

    best_meas = :sld_optimal
    best_cfi  = 0.0
    for (k, v) in results
        if v.cfi > best_cfi
            best_cfi  = v.cfi
            best_meas = k
        end
    end

    efficiency = Dict{Symbol, Float64}()
    for (k, v) in results
        efficiency[k] = qfi > 0 ? v.cfi / qfi : 0.0
    end

    return (qfi=qfi, results=results, best_measurement=best_meas,
            best_cfi=best_cfi, efficiency=efficiency)
end
