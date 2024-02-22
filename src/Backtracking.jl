#=
Backtracking line search implementation
=#

#=
Input parameters:
    - x: current iterate
    - g: gradient 
    - α: initial step size
    - p: descent direction
    - f: function evaluation 
=#

@with_kw mutable struct Backtracking{X, F, A} <: AbstractLineSearch
    x::X
    g::X
    p::X
    f::F
    α::A
    c1 = 1e-4
    β = 0.5
    iterations = 1_000
end

# Determine step size by backtracking line search
function Backtracking(backtrack::Backtracking)
    x = backtrack.x
    g = backtrack.g
    α = backtrack.α
    p = backtrack.p
    f = backtrack.f

    ϕ_0 = f(x)
    ϕ_α = f(x + α*p)
    iteration = 0

    while ϕ_α > ϕ_0 + c_1 * α * g * p
        # Increment the number of steps we've had to perform
        iteration += 1

        # Ensure termination
        if iteration > iterations
            throw(LineSearchException("Linesearch failed to converge, reached maximum iterations $(iterations).", α))
        end

        # Decrease the step-size
        α = β * α
    end

    return α
end