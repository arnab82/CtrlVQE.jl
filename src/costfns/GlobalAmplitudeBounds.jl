import ..CostFunctions
export GlobalAmplitudeBound

import ..Parameters, ..Devices, ..Signals

# NOTE: Implicitly use smooth bounding function.
wall(u) = exp(u - 1/u)
grad(u) = exp(u - 1/u) * (1 + 1/u^2)

"""
    GlobalAmplitudeBound(device, nsteps, dt, ΩMAX, λ, σ)

Smooth bounds on an integral over each drive signal in a device using RK4 for time evolution.

Each pulse is integrated separately; any "area" beyond ΩMAX is penalized.

# Parameters
- `device`: The device.
- `nsteps`: The number of time steps for RK4 integration.
- `dt`: Time step size for RK4.
- `ΩMAX`: Maximum allowable amplitude on a device.
- `λ`: Penalty strength.
- `σ`: Penalty effective width: smaller means steeper.

"""
struct GlobalAmplitudeBound{F,FΩ} <: CostFunctions.CostFunctionType{F}
    device::Devices.DeviceType{F,FΩ}
    nsteps::Int
    dt::F
    ΩMAX::F             # MAXIMUM PERMISSIBLE AMPLITUDE
    λ::F                # STRENGTH OF BOUND
    σ::F                # STEEPNESS OF BOUND

    function GlobalAmplitudeBound(
        device::Devices.DeviceType{DF,FΩ},
        nsteps::Int,
        dt::Real,
        ΩMAX::Real,
        λ::Real,
        σ::Real,
    ) where {DF,FΩ}
        F = promote_type(Float16, DF, real(FΩ), eltype(ΩMAX), eltype(λ), eltype(σ), dt)
        return new{F,FΩ}(device, nsteps, F(dt), ΩMAX, λ, σ)
    end
end

Base.length(fn::GlobalAmplitudeBound) = Parameters.count(fn.device)

function CostFunctions.cost_function(fn::GlobalAmplitudeBound{F, FΩ}) where {F, FΩ}
    dt = fn.dt
    nsteps = fn.nsteps
    Ω̄ = Vector{FΩ}(undef, nsteps)  # TO FILL, FOR EACH DRIVE

    Φ(t, Ω) = begin
        u = (abs(Ω) - fn.ΩMAX) / fn.σ
        return u ≤ 0 ? zero(u) : fn.λ * wall(u)
    end

    return (x̄) -> begin
        Parameters.bind(fn.device, x̄)
        total = zero(F)
        for i in 1:Devices.ndrives(fn.device)
            signal = Devices.drivesignal(fn.device, i)
            t = 0.0

            # Integrate using RK4 over the time steps
            for step in 1:nsteps
                Ω̄[step] = Signals.valueat(signal, t)
                total += Φ(t, Ω̄[step]) * dt
                t += dt
            end
        end
        return total
    end
end

function CostFunctions.grad_function_inplace(fn::GlobalAmplitudeBound{F, FΩ}) where {F, FΩ}
    dt = fn.dt
    nsteps = fn.nsteps
    Ω̄ = Vector{FΩ}(undef, nsteps)  # TO FILL, FOR EACH DRIVE
    ∂̄ = Vector{FΩ}(undef, nsteps)  # TO FILL, FOR EACH PARAMETER

    Φ(t, Ω, ∂) = begin
        u = (abs(Ω) - fn.ΩMAX) / fn.σ
        return u ≤ 0 ? zero(u) : fn.λ * grad(u) * real(conj(Ω) * ∂) / (abs(Ω) * fn.σ)
    end

    return (∇f̄, x̄) -> begin
        Parameters.bind(fn.device, x̄)
        ∇f̄ .= 0
        offset = 0
        for i in 1:Devices.ndrives(fn.device)
            signal = Devices.drivesignal(fn.device, i)
            t = 0.0
            L = Parameters.count(signal)

            for k in 1:L
                for step in 1:nsteps
                    Ω̄[step] = Signals.valueat(signal, t)
                    ∂̄[step] = Signals.partial(k, signal, t)
                    ∇f̄[offset + k] += Φ(t, Ω̄[step], ∂̄[step]) * dt
                    t += dt
                end
            end
            offset += L
        end
        return ∇f̄
    end
end
