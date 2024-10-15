import ..CostFunctions
export GlobalFrequencyBound

import ..Parameters, ..Devices, ..Signals

# NOTE: Implicitly use smooth bounding function.
wall(u) = exp(u - 1/u)
grad(u) = exp(u - 1/u) * (1 + 1/u^2)

"""
    GlobalFrequencyBound(device, ΔMAX, λ, σ)

Smooth bounds on the detuning frequencies of each drive signal in a device.

# Parameters
- `device`: The quantum device with drive signals.
- `ΔMAX`: Maximum allowable detuning.
- `λ`: Penalty strength.
- `σ`: Penalty effective width: smaller means steeper.

"""
struct GlobalFrequencyBound{F, FΩ} <: CostFunctions.CostFunctionType{F}
    device::Devices.DeviceType{F, FΩ}
    ΔMAX::F             # MAXIMUM PERMISSIBLE DETUNING
    λ::F                # STRENGTH OF BOUND
    σ::F                # STEEPNESS OF BOUND

    function GlobalFrequencyBound(
        device::Devices.DeviceType{DF, FΩ},
        ΔMAX::Real,
        λ::Real,
        σ::Real,
    ) where {DF, FΩ}
        F = promote_type(Float16, DF, real(FΩ), eltype(ΔMAX), eltype(λ), eltype(σ))
        return new{F, FΩ}(device, ΔMAX, λ, σ)
    end
end

Base.length(fn::GlobalFrequencyBound) = Parameters.count(fn.device)

function CostFunctions.cost_function(fn::GlobalFrequencyBound{F, FΩ}) where {F, FΩ}
    return (x̄) -> begin
        Parameters.bind(fn.device, x̄)
        total = zero(F)
        for i in 1:Devices.ndrives(fn.device)
            q = Devices.drivequbit(fn.device, i)
            Δ = Devices.detuningfrequency(fn.device, i, q)
            u = (abs(Δ) - fn.ΔMAX) / fn.σ
            total += u ≤ 0 ? zero(u) : fn.λ * wall(u)
        end
        return total
    end
end

function CostFunctions.grad_function_inplace(fn::GlobalFrequencyBound{F, FΩ}) where {F, FΩ}
    # Infer which parameters refer to drive frequencies
    nD = Devices.ndrives(fn.device)
    offset = 0
    for i in 1:nD
        signal = Devices.drivesignal(fn.device, i)
        offset += Parameters.count(signal)
    end

    if offset == length(fn)
        # No frequency parameters
        return (∇f̄, x̄) -> (∇f̄ .= 0; return ∇f̄)
    elseif offset + nD != length(fn)
        error("Ill-defined number of frequency parameters.")
    end

    # Assume x[offset + i] == ith frequency
    return (∇f̄, x̄) -> begin
        Parameters.bind(fn.device, x̄)
        ∇f̄ .= 0
        for i in 1:nD
            q = Devices.drivequbit(fn.device, i)
            Δ = Devices.detuningfrequency(fn.device, i, q)
            u = (abs(Δ) - fn.ΔMAX) / fn.σ
            ∇f̄[offset + i] += u ≤ 0 ? zero(u) : fn.λ * grad(u) * sign(Δ) / fn.σ
        end
        return ∇f̄
    end
end
