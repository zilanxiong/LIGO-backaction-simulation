module QuantumSensing

using QuantumToolbox
using LinearAlgebra

include("conversions.jl")
include("dynamics.jl")
include("sld.jl")
include("cfi.jl")
include("states.jl")

# GPU operator struct types — defined here so both CPU and GPU code can reference them.
# The constructors (build_mzi_operators, build_single_mode_operators) are in the CUDA extension.

export MZIOperators, SingleModeOperators
export build_mzi_operators, build_single_mode_operators

function build_mzi_operators end
function build_single_mode_operators end

struct MZIOperators
    N_basis::Int
    aC::QuantumObject
    aD::QuantumObject
    NC::QuantumObject
    ND::QuantumObject
    aA::QuantumObject
    aB::QuantumObject
    NA::QuantumObject
    NB::QuantumObject
    H_zero_two::QuantumObject
end

struct SingleModeOperators
    N_basis::Int
    a::QuantumObject
    n::QuantumObject
    H_zero::QuantumObject
end

end # module
