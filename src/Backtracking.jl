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

mutable struct Backtrack
    x::Vector{Float64}
    g::Vector{Float64}
    p::Vector{Float64}
    f::Function
    α::Float64
end

function suff_decrease(backtrack::Backtrack, c1::Float64 = 1e-4)
    # Setup (Function information)
    x = backtrack.x
    g = backtrack.g
    α = backtrack.α
    p = backtrack.p
    f = backtrack.f

    # Compute function value
    ϕ_0 = f(x)
    ϕ_α = f(x + α*p)

    # Backtracking line search Y/N
    is_success = false
    temp = g'p
    if ϕ_α > ϕ_0 + c1 * α * g'p
        is_success = true
    end

    return is_success
end

function Backtracking(x::Vector{Float64},g::Vector{Float64},p::Vector{Float64},f::Function,α::Float64)
    backtrack = Backtrack(x,g,p,f,α)

    # Setup (Parameters)
    c1 = 1e-4
    β = 0.5
    iterations = 1_000

    # Backtracking line search
    iteration = 0
    while suff_decrease(backtrack::Backtrack, c1::Float64)
        # Increment the number of steps we've had to perform
        iteration += 1

        # Exceed max iteration
        if iteration > iterations
            return NaN
        end

        # Decrease the step-size
        backtrack.α = β * backtrack.α
    end

    return backtrack.α
end


# Determine step size by backtracking line search
# function Backtracking(backtrack::Backtracking)
#     # Setup (Function information)
#     x = backtrack.x
#     g = backtrack.g
#     α = backtrack.α
#     p = backtrack.p
#     f = backtrack.f

#     # Setup (Parameters)
#     c1 = 1e-4
#     β = 0.5
#     iterations = 1_000

#     ϕ_0 = f(x)
#     ϕ_α = f(x + α*p)
#     iteration = 0

#     # Backtracking line search
#     while ϕ_α > ϕ_0 + c1 * α * g * p
#         # Increment the number of steps we've had to perform
#         iteration += 1

#         # Ensure termination
#         if iteration > iterations
#             # throw(LineSearchException("Linesearch failed to converge, reached maximum iterations $(iterations).", α))
#             return NaN
#         end

#         # Decrease the step-size
#         α = β * α

#         # Update function value at new iterate
#         ϕ_α = f(x + α*p)
#     end

#     return α
# end