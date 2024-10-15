import ..CostFunctions
export ProjectedEnergy

import ..LinearAlgebraTools, ..QubitOperators
import ..Parameters, ..Devices, ..Evolutions
import ..Bases, ..Operators

"""
    ProjectedEnergy(evolution, device, basis, frame, nsteps, dt, ψ0, O0; kwargs...)

Expectation value of a Hermitian observable.

The statevector is projected onto a binary logical space after time evolution,
    modeling an ideal quantum measurement where leakage is fully characterized.

# Arguments

- `evolution::Evolutions.EvolutionType`: the algorithm with which to evolve `ψ0`

- `device::Devices.DeviceType`: the device, which determines the time-evolution of `ψ0`

- `basis::Bases.BasisType`: the measurement basis

- `frame::Operators.StaticOperator`: the measurement frame

- `nsteps::Int`: Number of RK4 steps for time evolution.

- `dt::Float64`: Time step for each RK4 integration step.

- `ψ0`: the reference state, living in the physical Hilbert space of `device`.

- `O0`: a Hermitian matrix, living in the physical Hilbert space of `device`.

"""
struct ProjectedEnergy{F} <: CostFunctions.EnergyFunction{F}
    evolution::Evolutions.EvolutionType
    device::Devices.DeviceType
    basis::Bases.BasisType
    frame::Operators.StaticOperator
    nsteps::Int
    dt::Float64
    ψ0::Vector{Complex{F}}
    O0::Matrix{Complex{F}}

    function ProjectedEnergy(
        evolution::Evolutions.EvolutionType,
        device::Devices.DeviceType,
        basis::Bases.BasisType,
        frame::Operators.StaticOperator,
        nsteps::Int,
        dt::Float64,
        ψ0::AbstractVector,
        O0::AbstractMatrix,
    )
        # INFER FLOAT TYPE AND CONVERT ARGUMENTS
        F = real(promote_type(Float16, eltype(O0), eltype(ψ0), eltype(dt)))

        # CREATE OBJECT
        return new{F}(
            evolution, device, basis, frame, nsteps, dt,
            convert(Array{Complex{F}}, ψ0),
            convert(Array{Complex{F}}, O0),
        )
    end
end

Base.length(fn::ProjectedEnergy) = Parameters.count(fn.device)

function CostFunctions.trajectory_callback(
    fn::ProjectedEnergy,
    E::AbstractVector;
    callback=nothing
)
    workbasis = Evolutions.workbasis(fn.evolution)  # BASIS OF CALLBACK ψ
    U = Devices.basisrotation(fn.basis, workbasis, fn.device)
    π̄ = QubitOperators.localqubitprojectors(fn.device)
    ψ_ = similar(fn.ψ0)

    return (i, t, ψ) -> begin
        ψ_ .= ψ
        LinearAlgebraTools.rotate!(U, ψ_)  # ψ_ IS NOW IN MEASUREMENT BASIS
        LinearAlgebraTools.rotate!(π̄, ψ_)  # ψ_ IS NOW "MEASURED"
        # APPLY FRAME ROTATION TO STATE RATHER THAN OBSERVABLE
        Devices.evolve!(fn.frame, fn.device, fn.basis, -t, ψ_)
        E[i] = real(LinearAlgebraTools.expectation(fn.O0, ψ_))
        !isnothing(callback) && callback(i, t, ψ)
    end
end

function CostFunctions.cost_function(fn::ProjectedEnergy; callback=nothing)
    # DYNAMICALLY UPDATED STATEVECTOR
    ψ = copy(fn.ψ0)
    # OBSERVABLE, IN MEASUREMENT FRAME
    T = fn.nsteps * fn.dt
    OT = copy(fn.O0); Devices.evolve!(fn.frame, fn.device, fn.basis, T, OT)
    # INCLUDE PROJECTION ONTO COMPUTATIONAL SUBSPACE IN THE MEASUREMENT
    π̄ = QubitOperators.localqubitprojectors(fn.device)
    LinearAlgebraTools.rotate!(π̄, OT)

    return (x̄) -> begin
        Parameters.bind(fn.device, x̄)
        t = 0.0
        for step in 1:fn.nsteps
            ψ = rk4_step(fn.evolution, fn.device, fn.basis, t, ψ, fn.dt)
            callback !== nothing && callback(step, t, ψ)
            t += fn.dt
        end
        real(LinearAlgebraTools.expectation(OT, ψ))
    end
end

function CostFunctions.grad_function_inplace(fn::ProjectedEnergy{F}; ϕ=nothing) where {F}
    nsteps = fn.nsteps

    if isnothing(ϕ)
        return CostFunctions.grad_function_inplace(
            fn;
            ϕ=Array{F}(undef, nsteps + 1, Devices.ngrades(fn.device))
        )
    end

    # OBSERVABLE, IN MEASUREMENT FRAME
    T = nsteps * fn.dt
    OT = copy(fn.O0); Devices.evolve!(fn.frame, fn.device, fn.basis, T, OT)
    # INCLUDE PROJECTION ONTO COMPUTATIONAL SUBSPACE IN THE MEASUREMENT
    π̄ = QubitOperators.localqubitprojectors(fn.device)
    LinearAlgebraTools.rotate!(π̄, OT)

    return (∇f̄, x̄) -> begin
        Parameters.bind(fn.device, x̄)
        t = 0.0
        ψ = copy(fn.ψ0)
        for step in 1:nsteps
            ψ = rk4_step(fn.evolution, fn.device, fn.basis, t, ψ, fn.dt)
            t += fn.dt
        end

        Evolutions.gradientsignals(
            fn.evolution,
            fn.device,
            fn.basis,
            nsteps,
            fn.dt,
            fn.ψ0,
            OT;
            result=ϕ,  # Writes the gradient signal as needed.
        )
        ∇f̄ .= Devices.gradient(fn.device, nsteps, ϕ)
    end
end
