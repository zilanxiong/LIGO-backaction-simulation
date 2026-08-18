# CPU SLD and QFI calculation via eigendecomposition.

export calculate_sld, calculate_qfi, calculate_qfi_parallel

const prec_diff = 1e-5
const prec_eig  = 1e-8

"""
    calculate_sld(; dynamics, param_value, param_type, prec, prec1, kwargs...)

Calculate the Symmetric Logarithmic Derivative (SLD) operator on CPU.

Returns a NamedTuple with: L, eigenvalues_L, eigenvectors_L, qfi,
rho, eigenvalues_rho, eigenvectors_rho, drho.
"""
function calculate_sld(;
    dynamics::Function,
    param_value::Real=0.0,
    param_type::Symbol=:Δ,
    prec::Real=prec_diff,
    prec1::Real=prec_eig,
    kwargs...)

    params_plus = merge(kwargs, Dict(param_type => param_value + prec))
    params_minus = merge(kwargs, Dict(param_type => param_value - prec))

    function get_dense_dm(params)
        state = dynamics(; params...)
        if isket(state)
            state = ket2dm(state)
        end
        return Matrix(state.data)
    end

    ρ_plus = get_dense_dm(params_plus)
    ρ_minus = get_dense_dm(params_minus)
    dρ_mat = (ρ_plus - ρ_minus) / (2 * prec)

    params = merge(kwargs, Dict(param_type => param_value))
    ρ_mat = get_dense_dm(params)

    evals, evecs = eigen(Hermitian(ρ_mat))

    dρ_rot = evecs' * dρ_mat * evecs

    n = length(evals)
    L_rot = zeros(ComplexF64, n, n)

    for i in 1:n
        for j in 1:n
            denom = evals[i] + evals[j]
            if abs(denom) > prec1
                L_rot[i,j] = 2.0 * dρ_rot[i,j] / denom
            else
                L_rot[i,j] = 0.0
            end
        end
    end

    L_mat = evecs * L_rot * evecs'
    L_mat = (L_mat + L_mat') / 2.0

    qfi = real(tr(ρ_mat * L_mat^2))

    L_evals, L_evecs = eigen(Hermitian(L_mat))

    return (
        L = L_mat,
        eigenvalues_L = L_evals,
        eigenvectors_L = L_evecs,
        qfi = qfi,
        rho = ρ_mat,
        eigenvalues_rho = evals,
        eigenvectors_rho = evecs,
        drho = dρ_mat
    )
end


function calculate_qfi(; dynamics::Function=get_state_arms, param_value::Real=0.0,
                        param_type::Symbol=:Δ, prec::Real=prec_diff,
                        prec1::Real=prec_eig, kwargs...)
    params_plus = merge(kwargs, Dict(param_type => param_value + prec))
    params_minus = merge(kwargs, Dict(param_type => param_value - prec))

    function get_dense_dm(params)
        state = dynamics(; params...)
        if isket(state)
            state = ket2dm(state)
        end
        return Matrix(state.data)
    end

    ρ_plus = get_dense_dm(params_plus)
    ρ_minus = get_dense_dm(params_minus)
    dρ_mat = (ρ_plus - ρ_minus) / (2 * prec)

    params = merge(kwargs, Dict(param_type => param_value))
    ρ_mat = get_dense_dm(params)

    evals, evecs = eigen(Hermitian(ρ_mat))
    dρ_rot = evecs' * dρ_mat * evecs

    qfi = 0.0
    for i in 1:length(evals)
        for j in 1:length(evals)
            denom = evals[i] + evals[j]
            if abs(denom) > prec1
                qfi += 2.0 * abs2(dρ_rot[i,j]) / denom
            end
        end
    end

    real(qfi)
end

function calculate_qfi_parallel(; dynamics::Function, param_value::Real=0.0,
    param_type::Symbol=:Δ, prec::Real=prec_diff, prec1::Real=prec_eig, kwargs...)

    params_plus = merge(kwargs, Dict(param_type => param_value + prec))
    params_minus = merge(kwargs, Dict(param_type => param_value - prec))
    params_center = merge(kwargs, Dict(param_type => param_value))

    ρ_plus = Ref{Any}()
    ρ_minus = Ref{Any}()
    ρ_center = Ref{Any}()

    @sync begin
        Threads.@spawn ρ_plus[] = dynamics(; params_plus...)
        Threads.@spawn ρ_minus[] = dynamics(; params_minus...)
        Threads.@spawn ρ_center[] = dynamics(; params_center...)
    end

    ρ_c = isket(ρ_center[]) ? ket2dm(ρ_center[]) : ρ_center[]
    ρ_p = isket(ρ_plus[]) ? ket2dm(ρ_plus[]) : ρ_plus[]
    ρ_m = isket(ρ_minus[]) ? ket2dm(ρ_minus[]) : ρ_minus[]

    ρ_mat = Matrix{ComplexF64}(ρ_c.data)
    dρ_mat = Matrix{ComplexF64}((ρ_p.data - ρ_m.data) / (2 * prec))

    evals, evecs = eigen(Hermitian(ρ_mat))
    dρ_rot = evecs' * dρ_mat * evecs

    qfi = 0.0
    for i in eachindex(evals), j in eachindex(evals)
        denom = evals[i] + evals[j]
        if abs(denom) > prec1
            qfi += 2.0 * abs2(dρ_rot[i, j]) / denom
        end
    end

    real(qfi)
end
