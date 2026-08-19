module QuantumSensingCUDAExt

using QuantumSensing
using QuantumToolbox
using CUDA
using SparseArrays
using LinearAlgebra

import QuantumSensing: loss_to_kappa, phirms_to_chi,
    MZIOperators, SingleModeOperators

include("gpu_dynamics.jl")
include("gpu_sld.jl")

end # module
