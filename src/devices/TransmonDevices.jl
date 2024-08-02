import ..Parameters, ..Devices, ..LocallyDrivenDevices
export TransmonDevice, FixedFrequencyTransmonDevice

import ..LinearAlgebraTools
import ..Integrations, ..Signals

import ..Signals: SignalType
import ..LinearAlgebraTools: MatrixList
import ..Quples: Quple

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

using Memoization: @memoize
using LinearAlgebra: I, mul!

#=

The layout of this file includes a `TransmonDevice` interface,
    and a couple concrete types implementing it.

The purpose of the interface is to minimize code duplication for very similar devices,
    but it is all rather more complicated and ugly than it needs to be.
I suspect a better practice would be to implement each concrete type independently,
    definitely in its own file,
    and probably in its own module.

Therefore, I don't recommend looking too closely to this file as a model to emulate.

=#

abstract type AbstractTransmonDevice{F,FΩ} <: LocallyDrivenDevices.LocallyDrivenDevice{F} end

# THE INTERFACE TO IMPLEMENT

# Devices.nlevels
# Devices.nqubits
resonancefrequency(::AbstractTransmonDevice, q::Int)::Real = error("Not Implemented")
anharmonicity(::AbstractTransmonDevice, q::Int)::Real = error("Not Implemented")

ncouplings(::AbstractTransmonDevice)::Int = error("Not Implemented")
couplingpair(::AbstractTransmonDevice, k::Int)::Quple = error("Not Implemented")
couplingstrength(::AbstractTransmonDevice, k::Int)::Real = error("Not Implemented")

# Devices.ndrives
# LocallyDrivenDevices.drivequbit
drivefrequency(::AbstractTransmonDevice, i::Int)::Real = error("Not Implemented")
drivesignal(::AbstractTransmonDevice, i::Int)::SignalType = error("Not Implemented")

bindfrequencies(::AbstractTransmonDevice, ν̄::AbstractVector) = error("Not Implemented")


# THE INTERFACE ALREADY IMPLEMENTED

function Devices.noperators(device::AbstractTransmonDevice)
    return 1
end

function Devices.ngrades(device::AbstractTransmonDevice)
    return 2 * Devices.ndrives(device)
end

function LocallyDrivenDevices.gradequbit(device::AbstractTransmonDevice, j::Int)
    return LocallyDrivenDevices.drivequbit(device, ((j-1) >> 1) + 1)
end

function Devices.localalgebra(device::AbstractTransmonDevice; result=nothing)
    isnothing(result) && return Devices._localalgebra(device)

    m = Devices.nlevels(device)
    n = Devices.nqubits(device)

    a = zeros(eltype(device), (m,m))
    for i in 1:m-1
        a[i,i+1] = √i
    end

    for q in 1:n
        result[:,:,1,q] .= a
    end
    return result
end

function Devices.qubithamiltonian(
    device::AbstractTransmonDevice,
    ā,
    q::Int;
    result=nothing,
)
    a = @view(ā[:,:,1,q])
    Im = Matrix(I, size(a))     # UNAVOIDABLE ALLOCATION?

    result === nothing && (result = Matrix{eltype(a)}(undef, size(a)))
    result .= 0
    result .-= (anharmonicity(device,q)/2)  .* Im           #       - δ/2    I
    result = LinearAlgebraTools.rotate!(a', result)         #       - δ/2   a'a
    result .+= resonancefrequency(device,q) .* Im    # ω     - δ/2   a'a
    result = LinearAlgebraTools.rotate!(a', result)         # ω a'a - δ/2 a'a'aa
    return result
end

function Devices.staticcoupling(
    device::AbstractTransmonDevice,
    ā;
    result=nothing,
)
    d = size(ā,1)
    result === nothing && (result = Matrix{F}(undef, d, d))
    aTa = array(eltype(ā), size(result), LABEL)

    result .= 0
    for pq in 1:ncouplings(device)
        g = couplingstrength(device, pq)
        p, q = couplingpair(device, pq)

        aTa = mul!(aTa, (@view(ā[:,:,1,p]))', @view(ā[:,:,1,q]))
        result .+= g .* aTa
        result .+= g .* aTa'
    end
    return result
end

function Devices.driveoperator(
    device::AbstractTransmonDevice,
    ā,
    i::Int,
    t::Real;
    result=nothing,
)
    a = @view(ā[:,:,1,LocallyDrivenDevices.drivequbit(device, i)])
    e = exp(im * drivefrequency(device, i) * t)
    Ω = Signals.valueat(drivesignal(device, i), t)

    if result === nothing
        F = promote_type(eltype(a), eltype(e))  # Ω is no more complex than e.
        result = Matrix{F}(undef, size(a))
    end
    result .= 0

    result .+= (real(Ω) * e ) .* a
    result .+= (real(Ω) * e') .* a'

    if Ω isa Complex
        result .+= (imag(Ω) * im *e ) .* a
        result .+= (imag(Ω) * im'*e') .* a'
    end

    return result
end

function Devices.gradeoperator(
    device::AbstractTransmonDevice,
    ā,
    j::Int,
    t::Real;
    result=nothing,
)
    i = ((j-1) >> 1) + 1
    a = @view(ā[:,:,1,LocallyDrivenDevices.drivequbit(device, i)])
    e = exp(im * drivefrequency(device, i) * t)

    if result === nothing
        F = promote_type(eltype(a), eltype(e))
        result = Matrix{F}(undef, size(a))
    end
    result .= 0

    phase = Bool(j & 1) ? 1 : im    # Odd j -> "real" gradient operator; even j  -> "imag"
    result .+= (phase * e ) .* a
    result .+= (phase'* e') .* a'
    return result
end

function Devices.gradient(
    device::AbstractTransmonDevice{F,FΩ},
    grid::Integrations.IntegrationType,
    ϕ̄::AbstractMatrix;
    result=nothing,
) where {F,FΩ}
    L = Parameters.count(device)
    nD = Devices.ndrives(device)
    isnothing(result) && return Devices.gradient(
        device, grid, ϕ̄;
        result=Vector{F}(undef, L),
    )

    gradient_for_signals!(@view(result[1:L-nD]), device, grid, ϕ̄)
    gradient_for_frequencies!(@view(result[1+L-nD:L]), device, grid, ϕ̄)

    return result
end

function gradient_for_signals!(
    result::AbstractVector{F},
    device::AbstractTransmonDevice{F,FΩ},
    grid::Integrations.IntegrationType,
    ϕ̄::AbstractMatrix,
) where {F,FΩ}
    # CALCULATE GRADIENT FOR SIGNAL PARAMETERS
    t̄ = Integrations.lattice(grid)
    ∂̄ = array(FΩ, size(t̄), LABEL)
    Φ = (t, ∂, ϕα, ϕβ) -> (real(∂)*ϕα + imag(∂)*ϕβ)

    offset = 0
    for i in 1:Devices.ndrives(device)
        j = 2i - 1
        ϕ̄α = @view(ϕ̄[:,j])
        ϕ̄β = @view(ϕ̄[:,j+1])

        signal = drivesignal(device, i)
        L = Parameters.count(signal)

        for k in 1:L
            ∂̄ = Signals.partial(k, signal, t̄; result=∂̄)
            result[offset+k] = Integrations.integrate(grid, Φ, ∂̄, ϕ̄α, ϕ̄β)
        end
        offset += L
    end

    return result
end

function gradient_for_frequencies!(
    result::AbstractVector{F},
    device::AbstractTransmonDevice{F,FΩ},
    grid::Integrations.IntegrationType,
    ϕ̄::AbstractMatrix,
) where {F,FΩ}
    # CALCULATE GRADIENT FOR FREQUENCY PARAMETERS
    t̄ = Integrations.lattice(grid)
    Ω̄ = array(FΩ, size(t̄), LABEL)
    Φ = (t, Ω, ϕα, ϕβ) -> (t * (real(Ω)*ϕβ - imag(Ω)*ϕα))

    for i in 1:Devices.ndrives(device)
        j = 2i - 1
        ϕ̄α = @view(ϕ̄[:,j])
        ϕ̄β = @view(ϕ̄[:,j+1])

        signal = drivesignal(device, i)
        Ω̄ = Signals.valueat(signal, t̄; result=Ω̄)
        result[i] = Integrations.integrate(grid, Φ, Ω̄, ϕ̄α, ϕ̄β)
    end

    return result
end

function Parameters.count(device::AbstractTransmonDevice)
    cnt = Devices.ndrives(device)           # NOTE: There are `ndrives` frequencies.
    for i in 1:Devices.ndrives(device)
        cnt += Parameters.count(drivesignal(device, i))::Int
    end
    return cnt
end

function Parameters.names(device::AbstractTransmonDevice)
    names = []

    # STRING TOGETHER PARAMETER NAMES FOR EACH SIGNAL Ω̄[i]
    annotate(name,i) = "Ω$i(q$(device.q̄[i])):$name"
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        append!(names, (annotate(name,i) for name in Parameters.names(Ω)))
    end

    # TACK ON PARAMETER NAMES FOR EACH ν̄[i]
    append!(names, ("ν$i" for i in 1:Devices.ndrives(device)))
    return names
end

function Parameters.values(device::AbstractTransmonDevice{F,FΩ}) where {F,FΩ}
    values = F[]

    # STRING TOGETHER PARAMETERS FOR EACH SIGNAL Ω̄[i]
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        append!(values, Parameters.values(Ω)::Vector{F})
    end

    # TACK ON PARAMETERS FOR EACH ν̄[i]
    append!(values, (drivefrequency(device,i) for i in 1:Devices.ndrives(device)))
    return values
end

function Parameters.bind!(device::AbstractTransmonDevice, x̄::AbstractVector{F}) where {F}
    offset = 0

    # BIND PARAMETERS FOR EACH SIGNAL Ω̄[i]
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        L = Parameters.count(Ω)::Int
        Parameters.bind!(Ω, x̄[offset+1:offset+L])
        offset += L
    end

    # BIND PARAMETERS FOR EACH ν̄[i]
    bindfrequencies(device, x̄[offset+1:end])
end






"""
    TransmonDevice(ω̄, δ̄, ḡ, quples, q̄, ν̄, Ω̄, m)

A transmon device, modeling for example IBM's superconducting quantum computers.

Variational parameters include the shape parameters in each pulse,
    and the pulse frequencies.

# Arguments
- `ω̄`: a vector of angular frequencies specifying the resonance frequency of each qubit.
- `δ̄`: a vector of angular frequencies specifying the anharmonicity of each qubit.

- `ḡ`: a vector of angular frequencies specifying the strength of each coupling.
- `quples`: a vector of `Quple` identifying whcih qubits participate in each coupling.

- `q̄`: a vector of indices specifying the target qubit for each drive channel.
- `ν̄`: a vector of angular frequencies specifying the pulse frequencies for each channel.
- `Ω̄`: a vector of signals specifying the shape of the pulse for each channel.

- `m`: an integer specifying the number of physical levels to retain for each qubit.

"""
struct TransmonDevice{F,FΩ} <: AbstractTransmonDevice{F,FΩ}
    # QUBIT LISTS
    ω̄::Vector{F}
    δ̄::Vector{F}
    # COUPLING LISTS
    ḡ::Vector{F}
    quples::Vector{Quple}
    # DRIVE LISTS
    q̄::Vector{Int}
    ν̄::Vector{F}
    Ω̄::Vector{SignalType{F,FΩ}}
    # OTHER PARAMETERS
    m::Int

    function TransmonDevice(
        ω̄::AbstractVector{<:Real},
        δ̄::AbstractVector{<:Real},
        ḡ::AbstractVector{<:Real},
        quples::AbstractVector{Quple},
        q̄::AbstractVector{Int},
        ν̄::AbstractVector{<:AbstractFloat},
        Ω̄::AbstractVector{<:SignalType{F,FΩ}},
        m::Int,
    ) where {F,FΩ}
        # VALIDATE PARALLEL LISTS ARE CONSISTENT SIZE
        @assert length(ω̄) == length(δ̄) ≥ 1              # NUMBER OF QUBITS
        @assert length(ḡ) == length(quples)             # NUMBER OF COUPLINGS
        @assert length(q̄) == length(ν̄) == length(Ω̄)     # NUMBER OF DRIVES

        # VALIDATE QUBIT INDICES
        for (p,q) in quples
            @assert 1 <= p <= length(ω̄)
            @assert 1 <= q <= length(ω̄)
        end
        for q in q̄
            @assert 1 <= q <= length(ω̄)
        end

        # VALIDATE THAT THE HILBERT SPACE HAS SOME VOLUME...
        @assert m ≥ 2

        # STANDARDIZE TYPING
        return new{F,FΩ}(
            convert(Vector{F}, ω̄),
            convert(Vector{F}, δ̄),
            convert(Vector{F}, ḡ),
            quples,
            q̄,
            convert(Vector{F}, ν̄),
            [Ω for Ω in Ω̄],
            m,
        )
    end
end

Devices.nlevels(device::TransmonDevice) = device.m

Devices.nqubits(device::TransmonDevice) = length(device.ω̄)
resonancefrequency(device::TransmonDevice, q::Int) = device.ω̄[q]
anharmonicity(device::TransmonDevice, q::Int) = device.δ̄[q]

ncouplings(device::TransmonDevice) = length(device.quples)
couplingpair(device::TransmonDevice, k::Int) = device.quples[k]
couplingstrength(device::TransmonDevice, k::Int) = device.ḡ[k]

Devices.ndrives(device::TransmonDevice) = length(device.q̄)
LocallyDrivenDevices.drivequbit(device::TransmonDevice, i::Int) = device.q̄[i]
drivefrequency(device::TransmonDevice, i::Int) = device.ν̄[i]
drivesignal(device::TransmonDevice, i::Int) = device.Ω̄[i]

bindfrequencies(device::TransmonDevice, ν̄::AbstractVector) = (device.ν̄ .= ν̄)










"""
    FixedFrequencyTransmonDevice(ω̄, δ̄, ḡ, quples, q̄, ν̄, Ω̄, m)

A transmon device, modeling for example IBM's superconducting quantum computers.

Variational parameters include ONLY the shape parameters in each pulse.
Pulse frequencies are "frozen".

# Arguments
- `ω̄`: a vector of angular frequencies specifying the resonance frequency of each qubit.
- `δ̄`: a vector of angular frequencies specifying the anharmonicity of each qubit.

- `ḡ`: a vector of angular frequencies specifying the strength of each coupling.
- `quples`: a vector of `Quple` identifying whcih qubits participate in each coupling.

- `q̄`: a vector of indices specifying the target qubit for each drive channel.
- `ν̄`: a vector of angular frequencies specifying the pulse frequencies for each channel.
- `Ω̄`: a vector of signals specifying the shape of the pulse for each channel.

- `m`: an integer specifying the number of physical levels to retain for each qubit.

"""
struct FixedFrequencyTransmonDevice{F,FΩ} <: AbstractTransmonDevice{F,FΩ}
    # QUBIT LISTS
    ω̄::Vector{F}
    δ̄::Vector{F}
    # COUPLING LISTS
    ḡ::Vector{F}
    quples::Vector{Quple}
    # DRIVE LISTS
    q̄::Vector{Int}
    ν̄::Vector{F}
    Ω̄::Vector{SignalType{F,FΩ}}
    # OTHER PARAMETERS
    m::Int

    function FixedFrequencyTransmonDevice(
        ω̄::AbstractVector{<:Real},
        δ̄::AbstractVector{<:Real},
        ḡ::AbstractVector{<:Real},
        quples::AbstractVector{Quple},
        q̄::AbstractVector{Int},
        ν̄::AbstractVector{<:AbstractFloat},
        Ω̄::AbstractVector{<:SignalType{F,FΩ}},
        m::Int,
    ) where {F,FΩ}
        # VALIDATE PARALLEL LISTS ARE CONSISTENT SIZE
        @assert length(ω̄) == length(δ̄) ≥ 1              # NUMBER OF QUBITS
        @assert length(ḡ) == length(quples)             # NUMBER OF COUPLINGS
        @assert length(q̄) == length(ν̄) == length(Ω̄)     # NUMBER OF DRIVES

        # VALIDATE QUBIT INDICES
        for (p,q) in quples
            @assert 1 <= p <= length(ω̄)
            @assert 1 <= q <= length(ω̄)
        end
        for q in q̄
            @assert 1 <= q <= length(ω̄)
        end

        # VALIDATE THAT THE HILBERT SPACE HAS SOME VOLUME...
        @assert m ≥ 2

        # STANDARDIZE TYPING
        return new{F,FΩ}(
            convert(Vector{F}, ω̄),
            convert(Vector{F}, δ̄),
            convert(Vector{F}, ḡ),
            quples,
            q̄,
            convert(Vector{F}, ν̄),
            [Ω for Ω in Ω̄],
            m,
        )
    end
end

Devices.nlevels(device::FixedFrequencyTransmonDevice) = device.m

Devices.nqubits(device::FixedFrequencyTransmonDevice) = length(device.ω̄)
resonancefrequency(device::FixedFrequencyTransmonDevice, q::Int) = device.ω̄[q]
anharmonicity(device::FixedFrequencyTransmonDevice, q::Int) = device.δ̄[q]

ncouplings(device::FixedFrequencyTransmonDevice) = length(device.quples)
couplingpair(device::FixedFrequencyTransmonDevice, k::Int) = device.quples[k]
couplingstrength(device::FixedFrequencyTransmonDevice, k::Int) = device.ḡ[k]

Devices.ndrives(device::FixedFrequencyTransmonDevice) = length(device.q̄)
LocallyDrivenDevices.drivequbit(device::FixedFrequencyTransmonDevice, i::Int)=device.q̄[i]
drivefrequency(device::FixedFrequencyTransmonDevice, i::Int) = device.ν̄[i]
drivesignal(device::FixedFrequencyTransmonDevice, i::Int) = device.Ω̄[i]

bindfrequencies(device::FixedFrequencyTransmonDevice, ν̄::AbstractVector) = nothing


function Parameters.count(device::FixedFrequencyTransmonDevice)
    cnt = 0
    for i in 1:Devices.ndrives(device)
        cnt += Parameters.count(drivesignal(device, i))::Int
    end
    return cnt
end

function Parameters.names(device::FixedFrequencyTransmonDevice)
    names = []

    # STRING TOGETHER PARAMETER NAMES FOR EACH SIGNAL Ω̄[i]
    annotate(name,i) = "Ω$i(q$(device.q̄[i])):$name"
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        append!(names, (annotate(name,i) for name in Parameters.names(Ω)))
    end

    return names
end

function Parameters.values(device::FixedFrequencyTransmonDevice{F,FΩ}) where {F,FΩ}
    values = F[]

    # STRING TOGETHER PARAMETERS FOR EACH SIGNAL Ω̄[i]
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        append!(values, Parameters.values(Ω)::Vector{F})
    end

    return values
end

function Parameters.bind!(
    device::FixedFrequencyTransmonDevice,
    x̄::AbstractVector{F},
) where {F}
    offset = 0

    # BIND PARAMETERS FOR EACH SIGNAL Ω̄[i]
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        L = Parameters.count(Ω)::Int
        Parameters.bind!(Ω, x̄[offset+1:offset+L])
        offset += L
    end
end

function Devices.gradient(
    device::FixedFrequencyTransmonDevice{F,FΩ},
    grid::Integrations.IntegrationType,
    ϕ̄::AbstractMatrix;
    result=nothing,
) where {F,FΩ}
    L = Parameters.count(device)::Int
    isnothing(result) && return Devices.gradient(
        device, grid, ϕ̄;
        result=Vector{F}(undef, L),
    )

    gradient_for_signals!(result, device, grid, ϕ̄)

    return result
end









#= TODO (low): Other types

LegacyTransmonDevice: assume Ω(t) is real, no analytical gradient for ν, half as many grade operators.
LinearTransmonDevice: quples and ḡ replaced by n-length tuple ḡ, efficient static propagate.
TransmonDeviceSansRWA: implicitly one channel per qubit, different drive
a mix of the three I guess...

=#