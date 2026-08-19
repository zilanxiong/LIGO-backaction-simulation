# GPU SLD and QFI via CUDA eigendecomposition.

@inline function _to_dense_gpu(d)
    d isa CUDA.CUSPARSE.CuSparseMatrixCSC && return CuArray(d)
    d isa CUDA.CUSPARSE.CuSparseMatrixCSR && return CuArray(d)
    return d
end

@inline _param(sym::Symbol, val) = NamedTuple{(sym,)}((val,))

"""
    calculate_qfi(ops; dynamics, param_value, param_type, prec, prec1, kwargs...)

QFI via central-difference SLD formula, fully on GPU.
Returns a CPU Float64.
"""
function QuantumSensing.calculate_qfi(
    ops;
    dynamics::Function,
    param_value::Real = 0.0,
    param_type::Symbol = :Δ,
    prec::Real  = 1e-5,
    prec1::Real = 1e-8,
    kwargs...,
)
    θp = _param(param_type, param_value + prec)
    θm = _param(param_type, param_value - prec)
    θ0 = _param(param_type, param_value)

    ρ_plus   = dynamics(ops; kwargs..., θp...)
    ρ_minus  = dynamics(ops; kwargs..., θm...)
    ρ_center = dynamics(ops; kwargs..., θ0...)

    dρ_raw = (ρ_plus.data - ρ_minus.data) / (2 * prec)

    ρ_dense  = _to_dense_gpu(ρ_center.data)
    dρ_dense = _to_dense_gpu(dρ_raw)

    ρ_copy = copy(ρ_dense)
    if !(ρ_copy isa CuArray)
        error("GPU calculate_qfi received a CPU matrix ($(typeof(ρ_copy))). " *
              "The dynamics function returned a CPU state instead of GPU. " *
              "Check that the CUDA extension's get_state method is being dispatched.")
    end
    if eltype(ρ_copy) <: Complex
        evals, evecs = CUDA.CUSOLVER.heevd!('V', 'U', ρ_copy)
    else
        evals, evecs = CUDA.CUSOLVER.syevd!('V', 'U', ρ_copy)
    end

    dρ_rot = evecs' * dρ_dense * evecs

    n      = length(evals)
    evals_i = reshape(evals, n, 1)
    evals_j = reshape(evals, 1, n)
    denom  = evals_i .+ evals_j
    mask   = abs.(denom) .> prec1

    qfi_gpu = sum(2.0 .* abs2.(dρ_rot) ./ (denom .+ (1.0 .- mask)) .* mask)

    return qfi_gpu isa CUDA.CuArray ? Array(qfi_gpu)[1] : Float64(real(qfi_gpu))
end

"""
    calculate_sld(ops; dynamics, param_value, param_type, prec, prec1, kwargs...)

SLD operator and related quantities, on GPU. Returns CPU arrays.
"""
function QuantumSensing.calculate_sld(
    ops;
    dynamics::Function,
    param_value::Real = 0.0,
    param_type::Symbol = :Δ,
    prec::Real  = 1e-5,
    prec1::Real = 1e-8,
    kwargs...,
)
    θp = _param(param_type, param_value + prec)
    θm = _param(param_type, param_value - prec)
    θ0 = _param(param_type, param_value)

    ρ_plus   = dynamics(ops; kwargs..., θp...)
    ρ_minus  = dynamics(ops; kwargs..., θm...)
    ρ_center = dynamics(ops; kwargs..., θ0...)

    dρ_raw = (ρ_plus.data - ρ_minus.data) / (2 * prec)

    ρ_dense  = _to_dense_gpu(ρ_center.data)
    dρ_dense = _to_dense_gpu(dρ_raw)

    ρ_copy = copy(ρ_dense)
    if eltype(ρ_copy) <: Complex
        evals, evecs = CUDA.CUSOLVER.heevd!('V', 'U', ρ_copy)
    else
        evals, evecs = CUDA.CUSOLVER.syevd!('V', 'U', ρ_copy)
    end

    dρ_rot = evecs' * dρ_dense * evecs

    n      = length(evals)
    evals_i = reshape(evals, n, 1)
    evals_j = reshape(evals, 1, n)
    denom  = evals_i .+ evals_j
    mask   = abs.(denom) .> prec1

    L_rot = 2.0 .* dρ_rot ./ (denom .+ (1.0 .- mask)) .* mask

    L_gpu = evecs * L_rot * evecs'
    L_gpu = (L_gpu + L_gpu') / 2

    qfi_gpu = real(tr(ρ_dense * L_gpu^2))

    L_copy = copy(L_gpu)
    if eltype(L_copy) <: Complex
        L_evals, L_evecs = CUDA.CUSOLVER.heevd!('V', 'U', L_copy)
    else
        L_evals, L_evecs = CUDA.CUSOLVER.syevd!('V', 'U', L_copy)
    end

    return (
        L               = Matrix(L_gpu),
        eigenvalues_L   = Vector(L_evals),
        eigenvectors_L  = Matrix(L_evecs),
        qfi             = Float64(real(qfi_gpu)),
        rho             = Matrix(ρ_dense),
        eigenvalues_rho = Vector(evals),
        eigenvectors_rho = Matrix(evecs),
        drho            = Matrix(dρ_dense),
    )
end
