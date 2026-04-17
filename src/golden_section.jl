"""
    GoldenSection(; tol = 1e-7, maxiters = 100)

A derivative-free line search that minimizes a unimodal function by successively 
narrowing the interval containing the minimum using the golden ratio.
"""
@kwdef @concrete struct GoldenSection <: AbstractLineSearchAlgorithm
    tol = 1.0e-7
    maxiters::Int = 100
end

@concrete mutable struct GoldenSectionCache <: AbstractLineSearchCache
    ϕ
    f
    p
    u_cache
    fu_cache
    α
    φ
    resphi
    stats <: Union{SciMLBase.NLStats, Nothing}
    alg <: GoldenSection
    maxiters::Int
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::GoldenSection, fu, u;
        stats::Union{SciMLBase.NLStats, Nothing} = nothing, kwargs...
    )
    T = promote_type(eltype(fu), eltype(u))

    φ = (sqrt(T(5)) + 1) / 2
    resphi = 2 - φ 

    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)

    ϕ = @closure (f, p, u, du, α, u_cache, fu_cache) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(f, fu_cache, u_cache, p)
        add_nf!(stats)
        return @fastmath norm(fu_cache)^2 / 2
    end

    return GoldenSectionCache(
        ϕ, prob.f, prob.p, u_cache, fu_cache, T(1), φ, resphi, stats, alg, alg.maxiters
    )
end

function CommonSolve.solve!(cache::GoldenSectionCache, u, du)
    T = promote_type(eltype(du), eltype(u))
    ϕ = @closure α -> cache.ϕ(cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache)

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
    return LineSearchSolution(α_best, ReturnCode.Success)
end

function SciMLBase.reinit!(
        cache::GoldenSectionCache; p = missing, stats = missing, kwargs...
    )
    p !== missing && (cache.p = p)
    stats !== missing && (cache.stats = stats)
    cache.α = oftype(cache.α, true)
    return cache
end
