import ..Evolutions
import ..Evolutions: EvolutionType
export RK4

import ..LinearAlgebraTools
import ..Integrations, ..Devices
import ..Bases

import ..Bases: DRESSED
import ..Operators: STATIC, Drive

#import ..runge_kutta: rk4Integration

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

using LinearAlgebra: norm
"""
    rk4_step(evolution::EvolutionType, device::Devices.DeviceType, basis::Bases.BasisType, t::Float64, ψ::AbstractVector, dt::Float64)

The function `rk4_step` computes a single time step in the evolution of the quantum state `ψ` using the 4th-order Runge-Kutta (RK4) method.

## Derivation and RK4 Method:

The RK4 method approximates the solution of the differential equation that governs the time evolution of the quantum state `ψ(t)` under a time-dependent Hamiltonian `H(t)`:
    \frac{dψ(t)}{dt} = H(t) . ψ(t)
In RK4, the solution for `ψ(t + Δt)` is approximated as:
    ψ(t + Δt) = ψ(t) + \frac{1}{6} (k_1 + 2k_2 + 2k_3 + k_4)
where the `k` values are intermediate slopes (derivatives) evaluated at different points within the time step `Δt`.

### Step-by-Step Derivation:

1. **Initial derivative (k1)**:
   k_1 = f(t, ψ) = \frac{dψ(t)}{dt} = H(t) . ψ(t)
   This is the derivative at the start of the time step.
2. **Second derivative (k2)**:
   k_2 = f<(t + \frac{Δt}{2}, ψ + \frac{Δt}{2} k_1>)
   This is the derivative at the midpoint of the time step, using the state `ψ` advanced by half of the time step `Δt/2`.
3. **Third derivative (k3)**:
   k_3 = f<(t + \frac{Δt}{2}, ψ + \frac{Δt}{2} k_2>)
   Another midpoint derivative, but now using the updated state `ψ` from the `k2` calculation.
4. **Fourth derivative (k4)**:
   k_4 = f(t + Δt, ψ + Δt * k_3)
   The derivative at the full time step, using the state `ψ` advanced by the full time step `Δt`.
### Final Update to `ψ(t + Δt)`:
The new state `ψ(t + Δt)` is computed by combining these four derivatives:
    ψ(t + Δt) = ψ(t) + \frac{1}{6} <( k_1 + 2k_2 + 2k_3 + k_4 )>
This weighted average ensures high accuracy in the time evolution of the state.

### Arguments:
- `evolution`: The specific evolution method.
- `device`: The quantum device with a time-dependent Hamiltonian.
- `basis`: The basis in which the state is represented.
- `t`: The current time.
- `ψ`: The quantum state vector at time `t`.
- `dt`: The time step size `Δt`.

### Returns:
- The updated quantum state vector `ψ(t + Δt)` after applying the RK4 step.

"""

function rk4_step(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    t::Float64,
    ψ::AbstractVector{<:Complex{<:AbstractFloat}},
    dt::Float64
)
    # k1 = f(t, ψ)
    k1 = ψ_derivative(evolution, device, basis, t, ψ)

    # k2 = f(t + dt/2, ψ + dt/2 * k1)
    ψ_temp = ψ + 0.5 * dt * k1
    k2 = ψ_derivative(evolution, device, basis, t + 0.5 * dt, ψ_temp)

    # k3 = f(t + dt/2, ψ + dt/2 * k2)
    ψ_temp = ψ + 0.5 * dt * k2
    k3 = ψ_derivative(evolution, device, basis, t + 0.5 * dt, ψ_temp)

    # k4 = f(t + dt, ψ + dt * k3)
    ψ_temp = ψ + dt * k3
    k4 = ψ_derivative(evolution, device, basis, t + dt, ψ_temp)

    # Combine to get the new ψ: ψ(t + dt)
    ψ_new = ψ + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return ψ_new
end

"""
    ψ_derivative(evolution::EvolutionType, device::Devices.DeviceType, basis::Bases.BasisType, t::Float64, ψ::AbstractVector)

This function computes the time derivative of the quantum state `ψ(t)` at a given time `t`. The derivative is governed by the Schrödinger equation:
    \frac{dψ(t)}{dt} = H(t) . ψ(t)
where `H(t)` is the Hamiltonian of the system at time `t`.

### Derivation:
The time evolution of a quantum state is determined by the time-dependent Schrödinger equation:
    \frac{dψ(t)}{dt} = -i H(t) ψ(t)
In this function, `H(t)` represents the Hamiltonian at time `t`, which can include a static part (`STATIC`) and a time-dependent driving part (`Drive(t)`).

1. **Calculate the static part of the Hamiltonian**:
   - `U = Devices.evolver(STATIC, device, basis, t)`: This fetches the static component of the Hamiltonian in the desired basis.

2. **Calculate the time-dependent part of the Hamiltonian**:
   - `V = Devices.operator(Drive(t), device, basis)`: This computes the time-dependent operator that represents external driving fields at time `t`.

3. **Apply rotations for the interaction picture**:
   - `V = LinearAlgebraTools.rotate!(U', V)`: The driving Hamiltonian `V` is rotated into the correct basis using the static Hamiltonian.

4. **Compute the state derivative**:
   - Finally, the derivative of the quantum state `ψ` is calculated as `ψ'(t) = V * ψ(t)`.

### Arguments:
- `evolution`: The evolution method being used (e.g., `Direct`, `Toggle`, etc.).
- `device`: The quantum device specifying the Hamiltonian.
- `basis`: The basis in which the Hamiltonian and state are represented.
- `t`: The current time point.
- `ψ`: The quantum state vector at time `t`.

### Returns:
- The time derivative of the state `ψ(t)`, which is used for time integration methods such as RK4 or other time-stepping methods.
"""


# Function to calculate the time derivative of the quantum state ψ
function ψ_derivative(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    t::Float64,
    ψ::AbstractVector{<:Complex{<:AbstractFloat}}
)
    # Derivative of ψ under the device's Hamiltonian at time t
    # ψ'(t) = H(t) * ψ(t)
    U = Devices.evolver(STATIC, device, basis, t)
    V = Devices.operator(Drive(t), device, basis)
    V = LinearAlgebraTools.rotate!(U', V)
    ψ_deriv = LinearAlgebraTools.rotate!(V, ψ)
    return ψ_deriv
end



"""
    evolve!(evolution::EvolutionType, device::Devices.DeviceType, basis::Bases.BasisType, grid::IntegrationType{F}, ψ0::AbstractVector; kwargs...)

The `evolve!` function evolves the quantum state `ψ0` over time using the specified `evolution` method and the Hamiltonian of the `device`.
This evolution happens over a time grid specified by `grid`, and the state is updated in place.

### Derivation:
The function evolves the quantum state by solving the time-dependent Schrödinger equation:
    \frac{dψ(t)}{dt} = -i H(t) ψ(t)
Here, the Hamiltonian `H(t)` may have both static (`STATIC`) and time-dependent (`Drive(t)`) components. 
The evolution of the quantum state `ψ` is computed using a numerical integration method, such as the 4th-order Runge-Kutta (RK4).
### Key Steps:
1. **Time grid setup**:
   - The time evolution happens on a discrete grid of time points `t̄` obtained from `lattice(grid)`.
   - The total number of time steps is obtained from `nsteps(grid)`.

2. **Basis rotation**:
   - The quantum state `ψ0` may be represented in different bases (e.g., `OCCUPATION` or `DRESSED`). Before starting the evolution, `ψ0` is rotated into the working basis using `LinearAlgebraTools.rotate!`.

3. **Time evolution**:
   - The evolution is performed step by step over the grid using an integration method (e.g., Runge-Kutta 4th order). At each time step, the derivative of the state `ψ` is computed, and the state is updated accordingly.

4. **Callback function** (optional):
   - An optional `callback` function can be passed to monitor the state at each time step. The callback receives the current time step index, time, and quantum state `ψ`.

5. **Final basis rotation**:
   - After the time evolution is complete, the quantum state is rotated back into the original basis if it was initially rotated.

### Arguments:
- `evolution`: The evolution algorithm used to propagate the state. This defines how the quantum state evolves (e.g., using `Direct`, `Toggle`, etc.).
- `device`: The quantum device specifying the Hamiltonian to evolve under.
- `basis`: The basis in which the Hamiltonian and the state are represented. Defaults to the work basis of the evolution.
- `grid`: The time grid that defines the temporal lattice and step sizes (e.g., `TrapezoidalIntegration` or another type).
- `ψ0`: The initial quantum state vector. This state is evolved in place.
- `kwargs...`: Optional keyword arguments, including a `callback` function that is called at each iteration.

### Returns:
- The evolved quantum state `ψ(t)` after the time evolution has completed.
"""

# Modified evolve! function with RK4 method
function evolve!(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    grid::Integrations.IntegrationType,
    ψ0::AbstractVector;
    kwargs...
)
    # Get the temporal lattice and step sizes
    r = Integrations.nsteps(grid)
    t̄ = Integrations.lattice(grid)
    dt = Integrations.stepsize(grid)

    # Ensure the state ψ is in the working basis
    basis == workbasis(evolution) || (ψ0 = LinearAlgebraTools.rotate!(Devices.basisrotation(workbasis(evolution), basis, device), ψ0))

    # Evolution loop using Runge-Kutta 4th order method
    ψ = ψ0
    for i in 1:r
        # Perform a single RK4 step
        ψ = rk4_step(evolution, device, basis, t̄[i], ψ, dt)

        # Optional callback for tracking the evolution
        kwargs[:callback] !== nothing && kwargs[:callback](i, t̄[i], ψ)
    end

    # Rotate back into the given basis if necessary
    basis == workbasis(evolution) || (ψ = LinearAlgebraTools.rotate!(Devices.basisrotation(basis, workbasis(evolution), device), ψ))

    return ψ
end

"""
    rk4_step_gradientsignals(evolution::EvolutionType, device::Devices.DeviceType, basis::Bases.BasisType, t::Float64, ψ::AbstractVector, dt::Float64)
This function computes a single step in the evolution of the quantum state `ψ` using the 4th-order Runge-Kutta (RK4) method.

### Steps:
1. **Calculate k1**: 
   - `k1 = f(t, ψ)` represents the derivative of `ψ` at the current time `t` using the Hamiltonian at time `t`.

2. **Calculate k2**:
   - `k2 = f(t + Δt/2, ψ + Δt/2 * k1)` is the derivative at the midpoint, where `ψ` is advanced by half the time step `Δt/2`.

3. **Calculate k3**:
   - `k3 = f(t + Δt/2, ψ + Δt/2 * k2)` is another midpoint derivative, using `ψ` advanced with `k2`.

4. **Calculate k4**:
   - `k4 = f(t + Δt, ψ + Δt * k3)` is the derivative at the end of the time step, using `ψ` advanced by the full time step `Δt`.

5. **Final Update**:
   - The final updated state `ψ(t + Δt)` is computed as a weighted combination of the four derivatives:
   ψ(t + Δt) = ψ(t) + \frac{1}{6} (k_1 + 2k_2 + 2k_3 + k_4)

### Arguments:
- `evolution`: The evolution algorithm being used.
- `device`: The quantum device defining the Hamiltonian to evolve under.
- `basis`: The basis in which the Hamiltonian and state are represented.
- `t`: The current time.
- `ψ`: The quantum state at time `t`.
- `dt`: The time step size `Δt`.

### Returns:
- The updated quantum state `ψ(t + Δt)` after applying the RK4 step.
"""

function rk4_step_gradientsignals(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    t::Float64,
    ψ::AbstractVector{<:Complex{<:AbstractFloat}},
    dt::Float64
)
    # k1 = f(t, ψ)
    k1 = ψ_derivative(evolution, device, basis, t, ψ)

    # k2 = f(t + dt/2, ψ + dt/2 * k1)
    ψ_temp = ψ + 0.5 * dt * k1
    k2 = ψ_derivative(evolution, device, basis, t + 0.5 * dt, ψ_temp)

    # k3 = f(t + dt/2, ψ + dt/2 * k2)
    ψ_temp = ψ + 0.5 * dt * k2
    k3 = ψ_derivative(evolution, device, basis, t + 0.5 * dt, ψ_temp)

    # k4 = f(t + dt, ψ + dt * k3)
    ψ_temp = ψ + dt * k3
    k4 = ψ_derivative(evolution, device, basis, t + dt, ψ_temp)

    # Combine to get the new ψ: ψ(t + dt)
    ψ_new = ψ + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return ψ_new
end
function evolve!(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    nsteps::Int,
    dt::Float64,
    ψ0::AbstractVector;
    kwargs...
)
    # Ensure the state ψ is in the working basis
    basis == workbasis(evolution) || (ψ0 = LinearAlgebraTools.rotate!(Devices.basisrotation(workbasis(evolution), basis, device), ψ0))

    # Evolution loop using Runge-Kutta 4th order method
    ψ = ψ0
    t = 0.0
    for i in 1:nsteps
        # Perform a single RK4 step
        ψ = rk4_step(evolution, device, basis, t, ψ, dt)

        # Optional callback for tracking the evolution
        kwargs[:callback] !== nothing && kwargs[:callback](i, t, ψ)

        # Update time
        t += dt
    end

    # Rotate back into the given basis if necessary
    basis == workbasis(evolution) || (ψ = LinearAlgebraTools.rotate!(Devices.basisrotation(basis, workbasis(evolution), device), ψ))

    return ψ
end


"""
    gradientsignals(evolution::EvolutionType, device::Devices.DeviceType, basis::Bases.BasisType, grid::Integrations.IntegrationType, ψ0::AbstractVector, Ō::LinearAlgebraTools.MatrixList; result=nothing, callback=nothing)

This function computes the gradient signals for a quantum system by evolving the quantum state `ψ0` forward and the co-states `λ̄` backward in time using the Runge-Kutta 4th-order (RK4) method.

### Algorithm:
1. **Prepare the temporal lattice**:
   - The time grid `t̄` and step size `τ` are extracted from the integration grid.

2. **Allocate memory**:
   - Arrays are prepared for storing the quantum state `ψ`, co-states `λ̄`, and gradient signals `ϕ̄`.

3. **Forward evolution of `ψ` using RK4**:
   - The state `ψ` is evolved step by step from the initial time to the final time using the `rk4_step_gradientsignals` function for the forward propagation.

4. **Calculate co-states**:
   - After evolving `ψ`, co-states `λ̄` are initialized and rotated into the proper basis.

5. **Backward evolution of `λ` using RK4**:
   - The co-states `λ̄` are propagated backward in time from the final time to the initial time using RK4.

6. **Calculate gradient signals**:
   - At each time step, the braket between the gradient operator, co-states `λ̄`, and quantum state `ψ` is computed to produce the gradient signals `ϕ̄`.

### Arguments:
- `evolution`: The evolution algorithm being used (e.g., `Direct`, `Toggle`, etc.).
- `device`: The quantum device defining the Hamiltonian for the evolution.
- `basis`: The basis in which the quantum state `ψ` and Hamiltonian are represented.
- `grid`: The time grid for the evolution, defining time steps and bounds.
- `ψ0`: The initial quantum state vector to be evolved forward in time.
- `Ō`: A list of Hermitian observables for which gradients are computed.
- `result`: (Optional) A pre-allocated array to store gradient signals.
- `callback`: (Optional) A function that is called at each time step for custom tracking.

### Returns:
- A 3D array `ϕ̄[i,j,k]`, where each `ϕ̄[:,j,k]` is the gradient signal `ϕ_j(t)` evaluated at time `t` for observable `O_k`.
"""

# Modified gradientsignals function using RK4 for time evolution
function gradientsignals(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    grid::Integrations.IntegrationType,
    ψ0::AbstractVector,
    Ō::LinearAlgebraTools.MatrixList;
    result=nothing,
    callback=nothing,
)
    # PREPARE TEMPORAL LATTICE
    r = Integrations.nsteps(grid)
    τ = Integrations.stepsize(grid)
    t̄ = Integrations.lattice(grid)

    # PREPARE SIGNAL ARRAYS ϕ̄[i,j,k]
    if result === nothing
        F = real(LinearAlgebraTools.cis_type(ψ0))
        result = Array{F}(undef, r+1, Devices.ngrades(device), size(Ō,3))
    end

    # PREPARE STATE AND CO-STATES
    ψTYPE = LinearAlgebraTools.cis_type(ψ0)
    ψ = array(ψTYPE, size(ψ0), LABEL); ψ .= ψ0

    # Forward evolve ψ using RK4
    for i in 1:r
        ψ = rk4_step_gradientsignals(evolution, device, basis, t̄[i], ψ, τ)

        callback !== nothing && callback(i, t̄[i], ψ)
    end

    λ̄ = array(ψTYPE, (size(ψ0,1), size(Ō,3)), LABEL)
    for k in axes(Ō,3)
        λ̄[:,k] .= ψ
        LinearAlgebraTools.rotate!(@view(Ō[:,:,k]), @view(λ̄[:,k]))
    end

    # Rotate into OCCUPATION basis if necessary
    if basis != OCCUPATION
        U = Devices.basisrotation(OCCUPATION, basis, device)
        ψ = LinearAlgebraTools.rotate!(U, ψ)
        for k in axes(Ō,3)
            LinearAlgebraTools.rotate!(U, @view(λ̄[:,k]))
        end
    end

    # Calculate gradient signals at the last time step
    callback !== nothing && callback(r+1, t̄[r+1], ψ)
    for k in axes(Ō,3)
        λ = @view(λ̄[:,k])
        for j in 1:Devices.ngrades(device)
            z = Devices.braket(Gradient(j, t̄[end]), device, OCCUPATION, λ, ψ)
            result[r+1,j,k] = 2 * imag(z)   # ϕ̄[i,j,k] = -𝑖z + 𝑖z̄
        end
    end

    # Backward evolve λ using RK4
    for i in reverse(1:r)
        # Complete the previous time-step and start the next for both ψ and λ
        ψ = rk4_step_gradientsignals(evolution, device, OCCUPATION, t̄[i], ψ, -τ)
        for k in axes(Ō,3)
            λ = @view(λ̄[:,k])
            λ = rk4_step_gradientsignals(evolution, device, OCCUPATION, t̄[i], λ, -τ)
        end

        # Calculate gradient signal brakets
        callback !== nothing && callback(i, t̄[i], ψ)
        for k in axes(Ō,3)
            λ = @view(λ̄[:,k])
            for j in 1:Devices.ngrades(device)
                z = Devices.braket(Gradient(j, t̄[i]), device, OCCUPATION, λ, ψ)
                result[i,j,k] = 2 * imag(z) # ϕ̄[i,j,k] = -𝑖z + 𝑖z̄
            end
        end
    end

    return result
end
function gradientsignals(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    nsteps::Int,
    dt::Float64,
    ψ0::AbstractVector,
    Ō::LinearAlgebraTools.MatrixList;
    result=nothing,
    callback=nothing,
)
    # PREPARE SIGNAL ARRAYS ϕ̄[i,j,k]
    if result === nothing
        F = real(LinearAlgebraTools.cis_type(ψ0))
        result = Array{F}(undef, nsteps+1, Devices.ngrades(device), size(Ō,3))
    end

    # PREPARE STATE AND CO-STATES
    ψTYPE = LinearAlgebraTools.cis_type(ψ0)
    ψ = array(ψTYPE, size(ψ0), LABEL)
    ψ .= ψ0

    # Forward evolve ψ using RK4
    t = 0.0
    for i in 1:nsteps
        ψ = rk4_step_gradientsignals(evolution, device, basis, t, ψ, dt)

        callback !== nothing && callback(i, t, ψ)
        t += dt
    end

    λ̄ = array(ψTYPE, (size(ψ0,1), size(Ō,3)), LABEL)
    for k in axes(Ō,3)
        λ̄[:,k] .= ψ
        LinearAlgebraTools.rotate!(@view(Ō[:,:,k]), @view(λ̄[:,k]))
    end

    # Rotate into OCCUPATION basis if necessary
    if basis != OCCUPATION
        U = Devices.basisrotation(OCCUPATION, basis, device)
        ψ = LinearAlgebraTools.rotate!(U, ψ)
        for k in axes(Ō,3)
            LinearAlgebraTools.rotate!(U, @view(λ̄[:,k]))
        end
    end

    # Calculate gradient signals at the last time step
    callback !== nothing && callback(nsteps+1, t, ψ)
    for k in axes(Ō,3)
        λ = @view(λ̄[:,k])
        for j in 1:Devices.ngrades(device)
            z = Devices.braket(Gradient(j, t), device, OCCUPATION, λ, ψ)
            result[nsteps+1,j,k] = 2 * imag(z)   # ϕ̄[i,j,k] = -𝑖z + 𝑖z̄
        end
    end

    # Backward evolve λ using RK4
    for i in reverse(1:nsteps)
        # Complete the previous time-step and start the next for both ψ and λ
        t -= dt
        ψ = rk4_step_gradientsignals(evolution, device, OCCUPATION, t, ψ, -dt)
        for k in axes(Ō,3)
            λ = @view(λ̄[:,k])
            λ = rk4_step_gradientsignals(evolution, device, OCCUPATION, t, λ, -dt)
        end

        # Calculate gradient signal brakets
        callback !== nothing && callback(i, t, ψ)
        for k in axes(Ō,3)
            λ = @view(λ̄[:,k])
            for j in 1:Devices.ngrades(device)
                z = Devices.braket(Gradient(j, t), device, OCCUPATION, λ, ψ)
                result[i,j,k] = 2 * imag(z) # ϕ̄[i,j,k] = -𝑖z + 𝑖z̄
            end
        end
    end

    return result
end

