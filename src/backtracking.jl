"""
    BackTracking(; autodiff = nothing, c_1 = 1e-4, ρ_hi = 0.5, ρ_lo = 0.1,
        order = 3,
        maxstep = Inf, initial_alpha = true)

`BackTracking` line search algorithm based on the implementation in
[LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl).

`BackTracking` specifies a backtracking line-search that uses a quadratic or cubic
interpolant to determine the reduction in step-size.

E.g., if `f(α) > f(0) + c₁ α f'(0)`, then the quadratic interpolant of
`f(0)`, `f'(0)`, `f(α)` has a minimiser `α'` in the open interval `(0, α)`. More strongly,
there exists a factor ρ = ρ(c₁) such that `α' ≦ ρ α`.

This is a modification of the algorithm described in Nocedal Wright (2nd ed), Sec. 3.5.

`autodiff` is the automatic differentiation backend to use for the line search. This is only
used for the derivative of the objective function at the current step size. `autodiff` must
be specified if analytic jacobian/jvp/vjp is not available.

# Examples

```julia
using ADTypes
using LineSearch

alg = BackTracking(autodiff = AutoForwardDiff(), order = 3, maxiters = 100)
```
"""
@concrete struct BackTracking <: AbstractLineSearchAlgorithm
    autodiff
    c_1
    ρ_hi
    ρ_lo
    order <: Union{Val{2}, Val{3}}
    maxstep
    initial_alpha
    maxiters::Int
end

function BackTracking(;
        autodiff = nothing, c_1 = 1.0e-4, ρ_hi = 0.5, ρ_lo = 0.1,
        order::Union{Int, Val{2}, Val{3}} = 3, maxstep = Inf,
        initial_alpha = true, maxiters::Int = 1_000
    )
    order = order isa Val ? order : Val(order)
    @assert order isa Val{2} || order isa Val{3}
    return BackTracking(autodiff, c_1, ρ_hi, ρ_lo, order, maxstep, initial_alpha, maxiters)
end

@concrete mutable struct BackTrackingCache <: AbstractLineSearchCache
    merit_eval
    alpha
    initial_alpha
    alg <: BackTracking
    maxiters::Int
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::BackTracking, fu, u;
        stats::Union{SciMLBase.NLStats, Nothing} = nothing, autodiff = nothing, kwargs...
    )
    T = promote_type(eltype(fu), eltype(u))
    autodiff = autodiff !== nothing ? autodiff : alg.autodiff
    ev = init_merit(prob, fu, u; autodiff, stats)
    return build_backtracking_cache(ev, alg, u, T)
end

function CommonSolve.init(
        prob::OptimizationProblem, alg::BackTracking, u;
        stats::Union{SciMLBase.NLStats, Nothing} = nothing, kwargs...
    )
    ev = init_merit(prob, u; stats)
    return build_backtracking_cache(ev, alg, u, real(eltype(u)))
end

function CommonSolve.init(prob::OptimizationProblem, alg::BackTracking, gu, u; kwargs...)
    return CommonSolve.init(prob, alg, u; kwargs...)
end

function build_backtracking_cache(ev, alg::BackTracking, u, ::Type{T}) where {T}
    u_norm = @fastmath norm(u, Inf)
    alpha = min(alg.initial_alpha, alg.maxstep / u_norm)
    return BackTrackingCache(ev, T(alpha), T(alg.initial_alpha), alg, alg.maxiters)
end

function CommonSolve.solve!(
        cache::BackTrackingCache, u, du; ϕ0 = nothing, dϕ0 = nothing
    )
    T = promote_type(eltype(du), eltype(u))
    ev = cache.merit_eval
    invalidate!(ev)
    ϕ = @closure α -> merit_ϕ(ev, u, du, α)

    ϕ₀, dϕ₀ = merit_ϕdϕ_at_zero(ev, u, du, ϕ0, dϕ0)
    α₁, α₂ = cache.alpha, cache.alpha
    ϕx₀, ϕx₁ = ϕ₀, ϕ(α₁)

    finite_maxiters = -log2(eps(real(T)))
    iteration = 1
    while !isfinite(ϕx₁) && iteration ≤ finite_maxiters
        α₁ = α₂
        α₂ = α₁ / 2
        ϕx₁ = ϕ(α₂)
        iteration += 1
    end

    ϕx₁ ≤ ϕ₀ + T(cache.alg.c_1) * α₂ * dϕ₀ &&
        return LineSearchSolution(α₂, ReturnCode.Success)
    α_tmp = -(dϕ₀ * α₂^2) / (2 * (ϕx₁ - ϕ₀ - dϕ₀ * α₂))
    α₁ = α₂
    α_tmp = min(α_tmp, α₂ * T(cache.alg.ρ_hi))
    α₂ = max(α_tmp, α₂ * T(cache.alg.ρ_lo))
    ϕx₀, ϕx₁ = ϕx₁, ϕ(α₂)

    for _ in (iteration + 1):(cache.maxiters)
        ϕx₁ ≤ ϕ₀ + T(cache.alg.c_1) * α₂ * dϕ₀ &&
            return LineSearchSolution(α₂, ReturnCode.Success)

        α_tmp = compute_alpha_backtracking(cache.alg.order, T, dϕ₀, ϕ₀, ϕx₀, ϕx₁, α₁, α₂)

        α₁ = α₂
        α_tmp = min(α_tmp, α₂ * T(cache.alg.ρ_hi))
        α₂ = max(α_tmp, α₂ * T(cache.alg.ρ_lo))

        ϕx₀, ϕx₁ = ϕx₁, ϕ(α₂)
    end

    return LineSearchSolution(α₂, ReturnCode.Failure)
end

@inline function compute_alpha_backtracking(
        ::Val{2}, ::Type{T}, dϕ₀, ϕ₀, _, ϕx₁, _, α₂
    ) where {T}
    return -(dϕ₀ * α₂^2) / (2 * (ϕx₁ - ϕ₀ - dϕ₀ * α₂))
end

@inline function compute_alpha_backtracking(
        ::Val{3}, ::Type{T}, dϕ₀, ϕ₀, ϕx₀, ϕx₁, α₁, α₂
    ) where {T}
    div = inv(α₁^2 * α₂^2 * (α₂ - α₁))

    a₁ = α₁^2 * (ϕx₁ - ϕ₀ - dϕ₀ * α₂)
    a₂ = α₂^2 * (ϕx₀ - ϕ₀ - dϕ₀ * α₁)
    a = (a₁ - a₂) * div
    b = (-α₁ * a₁ + α₂ * a₂) * div

    return ifelse(iszero(a), dϕ₀ / 2b, (-b + sqrt(max(b^2 - 3a * dϕ₀, T(0)))) / 3a)
end

function SciMLBase.reinit!(
        cache::BackTrackingCache; p = missing, stats = missing, kwargs...
    )
    SciMLBase.reinit!(cache.merit_eval; p, stats)
    cache.alpha = cache.initial_alpha
    # NOTE: Don't zero out the stats here, since we don't own it
    return cache
end

set_initial_step!(cache::BackTrackingCache, α) = (cache.alpha = oftype(cache.alpha, α); cache)
