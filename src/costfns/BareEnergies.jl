import ..CostFunctions
export BareEnergy

import ..LinearAlgebraTools
import ..Parameters, ..Devices, ..Evolutions
import ..Bases, ..Operators

"""
    BareEnergy(evolution, device, basis, frame, nsteps, dt, ψ0, O0; kwargs...)

Expectation value of a Hermitian observable.

This type is called "bare" because it does not perform any projection steps
    (except perhaps what is hard-coded into the structure of the observable `O0`).

# Arguments

- `evolution::Evolutions.EvolutionType`: The algorithm with which to evolve `ψ0`.

- `device::Devices.DeviceType`: The device, which determines the time evolution of `ψ0`.

- `basis::Bases.BasisType`: The measurement basis. This also determines the basis which `ψ0` and `O0` are understood to be in.

- `frame::Operators.StaticOperator`: The measurement frame applied to `O0`.

- `nsteps::Int`: The number of time steps for the RK4 evolution.

- `dt::Float64`: The time step size for RK4.

- `ψ0`: The reference state, living in the physical Hilbert space of `device`.

- `O0`: A Hermitian matrix, living in the physical Hilbert space of `device`.

"""
struct BareEnergy{F} <: CostFunctions.EnergyFunction{F}
    evolution::Evolutions.EvolutionType
    device::Devices.DeviceType
    basis::Bases.BasisType
    frame::Operators.StaticOperator
    nsteps::Int
    dt::Float64
    ψ0::Vector{Complex{F}}
    O0::Matrix{Complex{F}}

    function BareEnergy(
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

Base.length(fn::BareEnergy) = Parameters.count(fn.device)

function CostFunctions.trajectory_callback(
    fn::BareEnergy,
    E::AbstractVector;
    callback=nothing
)
    workbasis = Evolutions.workbasis(fn.evolution)      # BASIS OF CALLBACK ψ
    U = Devices.basisrotation(fn.basis, workbasis, fn.device)
    ψ_ = similar(fn.ψ0)

    return (i, t, ψ) -> (
        ψ_ .= ψ;
        LinearAlgebraTools.rotate!(U, ψ_);  # ψ_ IS NOW IN MEASUREMENT BASIS
        # APPLY FRAME ROTATION TO STATE RATHER THAN OBSERVABLE
        Devices.evolve!(fn.frame, fn.device, fn.basis, -t, ψ_);
            # NOTE: Rotating observable only makes sense when time is always the same.
        E[i] = real(LinearAlgebraTools.expectation(fn.O0, ψ_));
        !isnothing(callback) && callback(i, t, ψ)
    )
end

function CostFunctions.cost_function(fn::BareEnergy; callback=nothing)
    # DYNAMICALLY UPDATED STATEVECTOR
    ψ = copy(fn.ψ0)
    # OBSERVABLE, IN MEASUREMENT FRAME
    T = fn.nsteps * fn.dt
    OT = copy(fn.O0)
    Devices.evolve!(fn.frame, fn.device, fn.basis, T, OT)

    return (x̄) -> begin
        Parameters.bind(fn.device, x̄)

        # Initialize evolution using RK4 integration
        t = 0.0

        # Perform RK4 evolution
        for i in 1:fn.nsteps
            ψ = rk4_step(fn.evolution, fn.device, fn.basis, t, ψ, fn.dt)
            if callback !== nothing
                callback(i, t, ψ)
            end
            t += fn.dt
        end

        # Calculate the expectation value
        return real(LinearAlgebraTools.expectation(OT, ψ))
    end
end

function CostFunctions.grad_function_inplace(fn::BareEnergy{F}; ϕ=nothing) where {F}
    if isnothing(ϕ)
        return CostFunctions.grad_function_inplace(
            fn;
            ϕ=Array{F}(undef, fn.nsteps + 1, Devices.ngrades(fn.device))
        )
    end

    # OBSERVABLE, IN MEASUREMENT FRAME
    T = fn.nsteps * fn.dt
    OT = copy(fn.O0); Devices.evolve!(fn.frame, fn.device, fn.basis, T, OT)

    return (∇f̄, x̄) -> (
        Parameters.bind(fn.device, x̄);
        
        # Perform gradient signals calculation using RK4 integration
        Evolutions.gradientsignals(
            fn.evolution,
            fn.device,
            fn.basis,
            fn.nsteps,
            fn.dt,
            fn.ψ0,
            OT;
            result=ϕ,
        );
        
        ∇f̄ .= Devices.gradient(fn.device, fn.nsteps, ϕ)
    )
end
