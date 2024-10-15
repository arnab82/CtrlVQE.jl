import ..CostFunctions
export AmplitudeBound

# NOTE: Use smooth bounding function.
wall(u) = exp(u - 1/u)
grad(u) = exp(u - 1/u) * (1 + 1/u^2)

"""
    AmplitudeBound(ΩMAX, λ, σ, L, Ω, paired)

Smooth bounds for explicitly windowed amplitude parameters.

# Parameters
- `ΩMAX`: Maximum allowable amplitude on a device.
- `λ`: Penalty strength.
- `σ`: Penalty effective width: smaller means steeper.
- `L`: Total number of parameters in cost function.
- `Ω`: Array of indices corresponding to amplitudes (only these are penalized).
- `paired`: Whether adjacent pairs of parameters give real+imaginary parts.

"""
struct AmplitudeBound{F} <: CostFunctions.CostFunctionType{F}
    ΩMAX::F             # MAXIMUM PERMISSIBLE AMPLITUDE
    λ::F                # STRENGTH OF BOUNDS
    σ::F                # STEEPNESS OF BOUNDS
    L::Int              # TOTAL NUMBER OF PARAMETERS IN COST FUNCTION
    Ω::Vector{Int}      # LIST OF INDICES THAT CORRESPOND TO AMPLITUDES
    paired::Bool        # FLAG THAT ADJACENT ITEMS IN Ω ARE REAL AND IMAGINARY PARTS

    function AmplitudeBound(
        ΩMAX::Real,
        λ::Real,
        σ::Real,
        L::Int,
        Ω::AbstractVector{Int},
        paired::Bool,
    )
        F = promote_type(Float16, eltype(ΩMAX), eltype(λ), eltype(σ))
        return new{F}(ΩMAX, λ, σ, L, convert(Vector{Int}, Ω), paired)
    end
end

Base.length(fn::AmplitudeBound) = fn.L

# Cost function to calculate the penalty
function CostFunctions.cost_function(fn::AmplitudeBound)
    if fn.paired
        # Ensure fn.Ω has an even number of elements if paired
        if length(fn.Ω) % 2 != 0
            error("Length of fn.Ω must be even when paired is true.")
        end
        Ωα = @view fn.Ω[1:2:end]
        Ωβ = @view fn.Ω[2:2:end]
    else
        Ωα = fn.Ω
    end

    return (x̄) -> begin
        total = 0
        for i in eachindex(Ωα)
            # Bounds checking
            if Ωα[i] > length(x̄) || (fn.paired && Ωβ[i] > length(x̄))
                error("Index out of bounds in the input parameter array x̄.")
            end

            α = x̄[Ωα[i]]
            β = fn.paired ? x̄[Ωβ[i]] : zero(α)
            r = sqrt(α^2 + β^2)

            # Avoid unnecessary calculations if r is zero
            if r != 0
                u = (r - fn.ΩMAX) / fn.σ
                if u > 0
                    total += fn.λ * wall(u)
                end
            end
        end
        total
    end
end

# Gradient function for the cost function
function CostFunctions.grad_function_inplace(fn::AmplitudeBound)
    if fn.paired
        # Ensure fn.Ω has an even number of elements if paired
        if length(fn.Ω) % 2 != 0
            error("Length of fn.Ω must be even when paired is true.")
        end
        Ωα = @view fn.Ω[1:2:end]
        Ωβ = @view fn.Ω[2:2:end]
    else
        Ωα = fn.Ω
    end

    return (∇f̄, x̄) -> begin
        ∇f̄ .= 0  # Reset gradients
        for i in eachindex(Ωα)
            # Bounds checking
            if Ωα[i] > length(x̄) || (fn.paired && Ωβ[i] > length(x̄))
                error("Index out of bounds in the input parameter array x̄.")
            end

            α = x̄[Ωα[i]]
            β = fn.paired ? x̄[Ωβ[i]] : zero(α)
            r = sqrt(α^2 + β^2)

            # Avoid division by zero
            if r != 0
                u = (r - fn.ΩMAX) / fn.σ
                if u > 0
                    grad_u = fn.λ * grad(u) / fn.σ
                    ∇f̄[Ωα[i]] += grad_u * (α / r)
                    if fn.paired
                        ∇f̄[Ωβ[i]] += grad_u * (β / r)
                    end
                end
            end
        end
        ∇f̄
    end
end
