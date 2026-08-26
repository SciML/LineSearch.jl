"""
    GoldenSection(; tol = 1e-7, maxiters = 100)

A derivative-free line search that minimizes a unimodal merit function by
successively narrowing the interval containing the minimum using the golden
ratio.

# Keyword Arguments

- `tol`: interval-width tolerance used to stop the golden-section search.
- `maxiters`: maximum number of interval-refinement iterations.

# Examples

```julia
using LineSearch

alg = GoldenSection(tol = 1e-8, maxiters = 200)
```
"""
@kwdef @concrete struct GoldenSection <: AbstractLineSearchAlgorithm
    tol = 1.0e-7
    maxiters::Int = 100
end

@concrete mutable struct GoldenSectionCache <: AbstractLineSearchCache
    merit_eval
    α
    φ
    resphi
    alg <: GoldenSection
    maxiters::Int
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::GoldenSection, fu, u;
        stats::Union{SciMLBase.NLStats, Nothing} = nothing, kwargs...
    )
    T = promote_type(eltype(fu), eltype(u))
    # Derivative free: no Jacobian operator is needed.
    ev = init_merit(prob, fu, u; stats, need_deriv = false)
    return build_golden_section_cache(ev, alg, T)
end

function CommonSolve.init(
        prob::OptimizationProblem, alg::GoldenSection, u;
        stats::Union{SciMLBase.NLStats, Nothing} = nothing, kwargs...
    )
    ev = init_merit(prob, u; stats, need_deriv = false)
    return build_golden_section_cache(ev, alg, real(eltype(u)))
end

function CommonSolve.init(prob::OptimizationProblem, alg::GoldenSection, gu, u; kwargs...)
    return CommonSolve.init(prob, alg, u; kwargs...)
end

function build_golden_section_cache(ev, alg::GoldenSection, ::Type{T}) where {T}
    φ = (sqrt(T(5)) + 1) / 2
    return GoldenSectionCache(ev, T(1), φ, 2 - φ, alg, alg.maxiters)
end

function CommonSolve.solve!(cache::GoldenSectionCache, u, du)
    T = promote_type(eltype(du), eltype(u))
    ev = cache.merit_eval
    invalidate!(ev)
    ϕ = @closure α -> merit_ϕ(ev, u, du, α)

    a, b = zero(T), T(cache.α)

    x1 = a + cache.resphi * (b - a)
    x2 = b - cache.resphi * (b - a)
    f1, f2 = ϕ(x1), ϕ(x2)

    for _ in 1:(cache.maxiters)
        abs(b - a) ≤ cache.alg.tol && break
        if f1 < f2
            b = x2;  x2 = x1;  f2 = f1
            x1 = a + cache.resphi * (b - a);  f1 = ϕ(x1)
        else
            a = x1;  x1 = x2;  f1 = f2
            x2 = b - cache.resphi * (b - a);  f2 = ϕ(x2)
        end
    end

    α_best = (a + b) / 2
    ϕ_best = ensure_value_at!(ev, u, du, α_best)
    return LineSearchSolution(α_best, ReturnCode.Success, ϕ_best, nothing)
end

function SciMLBase.reinit!(
        cache::GoldenSectionCache; p = missing, stats = missing, kwargs...
    )
    SciMLBase.reinit!(cache.merit_eval; p, stats)
    cache.α = oftype(cache.α, true)
    return cache
end
