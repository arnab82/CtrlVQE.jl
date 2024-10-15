import ..CostFunctions
export Normalization

import ..LinearAlgebraTools, ..QubitOperators
import ..Parameters, ..Devices, ..Evolutions
import ..Bases, ..Operators

"""
    Normalization(evolution, device, basis, nsteps, dt, ψ0; kwargs...)

The norm of a state vector in a binary logical space.

# Arguments

- `evolution::Evolutions.EvolutionType`: the algorithm with which to evolve `ψ0`.

- `device::Devices.DeviceType`: the device, which determines the time evolution of `ψ0`.

- `basis::Bases.BasisType`: the measurement basis. Also determines the basis which `ψ0` is understood to be given in.
  An intuitive choice is `Bases.OCCUPATION`, aka. the qubits' Z basis. If you change this argument, note that you may want to rotate `ψ0`.

- `nsteps`: Number of steps for time evolution.

- `dt`: Time step size.

- `ψ0`: The reference state, living in the physical Hilbert space of `device`.

"""
struct Normalization{F} <: CostFunctions.EnergyFunction{F}
    evolution::Evolutions.EvolutionType
    device::Devices.DeviceType
    basis::Bases.BasisType
    nsteps::Int
    dt::F
    ψ0::Vector{Complex{F}}

    function Normalization(
        evolution::Evolutions.EvolutionType,
        device::Devices.DeviceType,
        basis::Bases.BasisType,
        nsteps::Int,
        dt::Real,
        ψ0::AbstractVector,
    )
        # Infer float type and convert arguments
        F = promote_type(Float16, eltype(ψ0), dt)

        # Create object
        return new{F}(evolution, device, basis, nsteps, F(dt), convert(Array{Complex{F}}, ψ0))
    end
end

Base.length(fn::Normalization) = Parameters.count(fn.device)

function CostFunctions.trajectory_callback(
    fn::Normalization,
    F::AbstractVector;
    callback=nothing
)
    workbasis = Evolutions.workbasis(fn.evolution)  # Basis of callback ψ
    U = Devices.basisrotation(fn.basis, workbasis, fn.device)
    π̄ = QubitOperators.localqubitprojectors(fn.device)
    ψ_ = similar(fn.ψ0)

    return (i, t, ψ) -> begin
        ψ_ .= ψ
        LinearAlgebraTools.rotate!(U, ψ_)  # ψ_ is now in the measurement basis
        F[i] = real(LinearAlgebraTools.expectation(π̄, ψ_))
        !isnothing(callback) && callback(i, t, ψ)
    end
end


function CostFunctions.cost_function(fn::Normalization; callback=nothing)
    # Dynamically updated state vector
    ψ = copy(fn.ψ0)
    π̄ = QubitOperators.localqubitprojectors(fn.device)

    return (x̄) -> begin
        Parameters.bind(fn.device, x̄)
        t = 0.0

        # Evolve using RK4
        for i in 1:fn.nsteps
            ψ = rk4_step(fn.evolution, fn.device, fn.basis, t, ψ, fn.dt)
            callback !== nothing && callback(i, t, ψ)
            t += fn.dt
        end

        return real(LinearAlgebraTools.expectation(π̄, ψ))
    end
end

function CostFunctions.grad_function_inplace(fn::Normalization{F}; ϕ=nothing) where {F}
    if isnothing(ϕ)
        return CostFunctions.grad_function_inplace(
            fn;
            ϕ=Array{F}(undef, fn.nsteps + 1, Devices.ngrades(fn.device))
        )
    end

    # Observable - It's the projection operator
    Π = QubitOperators.qubitprojector(fn.device)

    return (∇f̄, x̄) -> begin
        Parameters.bind(fn.device, x̄)
        Evolutions.gradientsignals(
            fn.evolution,
            fn.device,
            fn.basis,
            fn.nsteps,
            fn.dt,
            fn.ψ0,
            Π;
            result=ϕ  # This writes the gradient signal as needed.
        )
        ∇f̄ .= Devices.gradient(fn.device, fn.nsteps, ϕ)
        return ∇f̄
    end
end
