"""
    HagerZhangLineSearch(; autodiff = nothing, δ = 0.1, σ = 0.9, ε = 1e-6,
        θ = 0.5, ρ = 5.0, maxiters = 50, α_init = 1.0, α_max = Inf)

Hager–Zhang line search (Hager & Zhang, *SIAM J. Optim.* 16(1), 2005;
CG_DESCENT, ACM TOMS 32(1), 2006).

Accepts a step satisfying either the original Wolfe conditions or the
*approximate* Wolfe conditions

```
σ ϕ'(0) ≤ ϕ'(α) ≤ (2δ - 1) ϕ'(0),    ϕ(α) ≤ ϕ(0) + ε_k
```

The approximate conditions stay testable once `ϕ(α) - ϕ(0)` has fallen to the
level of floating-point roundoff, which is exactly where the original conditions
become unsatisfiable — near a minimizer. A search restricted to the original
conditions stalls there, above the gradient floor.

Works with both merits: pass a `NonlinearProblem` for `‖F‖²/2` or an
`OptimizationProblem` for the objective itself.

# Keyword Arguments

- `autodiff`: AD backend for the directional derivative. Only consulted for the
  residual merit, which needs a Jacobian-vector product; the objective merit
  uses the gradient directly.
- `δ`: sufficient-decrease coefficient, `0 < δ < 1/2`.
- `σ`: curvature coefficient, `δ ≤ σ < 1`.
- `ε`: sets `ε_k = ε |ϕ(0)|`, the roundoff-level slack in the approximate
  conditions.
- `θ`: bisection weight used when an interval must be shrunk.
- `ρ`: expansion factor used while bracketing.
- `maxiters`: cap on outer iterations, bracket expansions, and bisections.
- `α_init`, `α_max`: initial and maximum step length.

# Examples

```julia
using LineSearch

alg = HagerZhangLineSearch(δ = 0.1, σ = 0.9)
```
"""
@kwdef @concrete struct HagerZhangLineSearch <: AbstractLineSearchAlgorithm
    autodiff = nothing
    δ = 0.1
    σ = 0.9
    ε = 1.0e-6
    θ = 0.5
    ρ = 5.0
    maxiters::Int = 50
    α_init = 1.0
    α_max = Inf
end

@concrete mutable struct HagerZhangLineSearchCache <: AbstractLineSearchCache
    merit_eval
    α_init
    α_max
    δ
    σ
    ε
    θ
    ρ
    maxiters::Int
    alg <: HagerZhangLineSearch
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::HagerZhangLineSearch, fu, u;
        stats::Union{SciMLBase.NLStats, Nothing} = nothing, autodiff = nothing, kwargs...
    )
    T = promote_type(eltype(fu), eltype(u))
    autodiff = autodiff !== nothing ? autodiff : alg.autodiff
    ev = init_merit(prob, fu, u; autodiff, stats)
    return build_hz_cache(ev, alg, T)
end

function CommonSolve.init(
        prob::OptimizationProblem, alg::HagerZhangLineSearch, u;
        stats::Union{SciMLBase.NLStats, Nothing} = nothing, kwargs...
    )
    T = real(eltype(u))
    ev = init_merit(prob, u; stats)
    return build_hz_cache(ev, alg, T)
end

function CommonSolve.init(
        prob::OptimizationProblem, alg::HagerZhangLineSearch, gu, u; kwargs...
    )
    return CommonSolve.init(prob, alg, u; kwargs...)
end

function build_hz_cache(ev, alg::HagerZhangLineSearch, ::Type{T}) where {T}
    return HagerZhangLineSearchCache(
        ev, T(alg.α_init), T(alg.α_max), T(alg.δ), T(alg.σ),
        T(alg.ε), T(alg.θ), T(alg.ρ), alg.maxiters, alg
    )
end

# A sampled point of ϕ. Immutable so brackets are carried without allocating.
struct HZPoint{T}
    α::T
    ϕ::T
    dϕ::T
end

@inline hz_finite(p::HZPoint) = isfinite(p.ϕ) && isfinite(p.dϕ)

struct HZParams{T}
    ϕ0::T
    dϕ0::T
    ε_k::T
    δ::T
    σ::T
end

@inline function hz_wolfe(pt::HZPoint{T}, p::HZParams{T}) where {T}
    hz_finite(pt) || return false
    orig = (pt.ϕ - p.ϕ0 ≤ p.δ * pt.α * p.dϕ0) && (pt.dϕ ≥ p.σ * p.dϕ0)
    orig && return true
    return (pt.ϕ ≤ p.ϕ0 + p.ε_k) &&
        (pt.dϕ ≥ p.σ * p.dϕ0) &&
        (pt.dϕ ≤ (2 * p.δ - one(T)) * p.dϕ0)
end

# Bracket collapse is judged relative to the bracket's own magnitude, never to
# `1`. With a large gradient the optimal step is legitimately far below eps(T),
# and an absolute floor there would end the search before it began.
@inline function hz_degenerate(a::HZPoint{T}, b::HZPoint{T}) where {T}
    w = b.α - a.α
    return !(w > eps(T) * max(abs(b.α), floatmin(T)))
end

@inline function hz_secant(a::HZPoint{T}, b::HZPoint{T}) where {T}
    den = b.dϕ - a.dϕ
    den == 0 && return T(NaN)
    return (a.α * b.dϕ - b.α * a.dϕ) / den
end

# Procedure U3: ϕ'(c) < 0 while ϕ(c) has risen above ϕ0 + ε_k.
function hz_bisect(
        p::HZParams{T}, ϕdϕ, a_lo::HZPoint{T}, a_hi::HZPoint{T},
        θ::T, maxiters::Int
    ) where {T}
    â, b̂ = a_lo, a_hi
    for _ in 1:maxiters
        d = ϕdϕ((one(T) - θ) * â.α + θ * b̂.α)
        hz_wolfe(d, p) && return (â, b̂, true, d)
        if !hz_finite(d)
            b̂ = d
            continue
        end
        if d.dϕ ≥ zero(T)
            return (â, d, false, â)
        elseif d.ϕ ≤ p.ϕ0 + p.ε_k
            â = d
        else
            b̂ = d
        end
        hz_degenerate(â, b̂) && return (â, b̂, false, â)
    end
    return (â, b̂, false, â)
end

# Procedure U: refine the bracket [a, b] with the probe c.
function hz_update(
        p::HZParams{T}, ϕdϕ, a::HZPoint{T}, b::HZPoint{T},
        c::HZPoint{T}, θ::T, maxiters::Int
    ) where {T}
    (!isfinite(c.α) || c.α ≤ a.α || c.α ≥ b.α) && return (a, b, false, a)
    hz_finite(c) || return hz_bisect(p, ϕdϕ, a, c, θ, maxiters)
    c.dϕ ≥ zero(T) && return (a, c, false, a)
    c.ϕ ≤ p.ϕ0 + p.ε_k && return (c, b, false, c)
    return hz_bisect(p, ϕdϕ, a, c, θ, maxiters)
end

# Procedure secant²: two secant steps give the superlinear interval reduction
# that makes this search cheap in function evaluations.
function hz_secant2(
        p::HZParams{T}, ϕdϕ, a::HZPoint{T}, b::HZPoint{T},
        θ::T, maxiters::Int
    ) where {T}
    c = hz_secant(a, b)
    (!isfinite(c) || c ≤ a.α || c ≥ b.α) && (c = (a.α + b.α) / 2)
    cpt = ϕdϕ(c)
    hz_wolfe(cpt, p) && return (a, b, true, cpt)
    A, B, found, best = hz_update(p, ϕdϕ, a, b, cpt, θ, maxiters)
    found && return (A, B, true, best)

    c̄ = if cpt.α == B.α
        hz_secant(b, B)
    elseif cpt.α == A.α
        hz_secant(a, A)
    else
        return (A, B, false, best)
    end

    if isfinite(c̄) && A.α < c̄ < B.α
        c̄pt = ϕdϕ(c̄)
        hz_wolfe(c̄pt, p) && return (A, B, true, c̄pt)
        A, B, found, best = hz_update(p, ϕdϕ, A, B, c̄pt, θ, maxiters)
        found && return (A, B, true, best)
    end
    return (A, B, false, best)
end

# Procedure B: expand until [a, b] brackets a minimizer.
function hz_bracket(
        p::HZParams{T}, ϕdϕ, c0::HZPoint{T}, zero_pt::HZPoint{T},
        θ::T, ρ::T, α_max::T, maxiters::Int
    ) where {T}
    a = zero_pt
    cj = c0
    for _ in 1:maxiters
        if !hz_finite(cj)
            A, B, found, best = hz_bisect(p, ϕdϕ, a, cj, θ, maxiters)
            return (A, B, found, best, true)
        end
        if cj.dϕ ≥ zero(T)
            return (a, cj, false, a, true)
        elseif cj.ϕ > p.ϕ0 + p.ε_k
            A, B, found, best = hz_bisect(p, ϕdϕ, zero_pt, cj, θ, maxiters)
            return (A, B, found, best, true)
        end
        a = cj
        αnext = ρ * cj.α
        if αnext ≥ α_max
            αnext = α_max
            αnext ≤ cj.α && return (a, cj, false, a, false)
        end
        cj = ϕdϕ(αnext)
        hz_wolfe(cj, p) && return (a, cj, true, cj, true)
    end
    return (a, cj, false, a, false)
end

@inline function hz_best_effort(p::HZParams{T}, a::HZPoint{T}, b::HZPoint{T}) where {T}
    hz_finite(a) && a.ϕ < p.ϕ0 && return a
    hz_finite(b) && b.ϕ < p.ϕ0 && return b
    return HZPoint(zero(T), p.ϕ0, p.dϕ0)
end

function CommonSolve.solve!(
        cache::HagerZhangLineSearchCache, u, du; ϕ0 = nothing, dϕ0 = nothing
    )
    T = promote_type(eltype(du), eltype(u))
    ev = cache.merit_eval
    ϕdϕ = @closure α -> begin
        ϕ, dϕ = merit_ϕdϕ(ev, u, du, α)
        # A non-finite value carries no usable slope; report it as "overshot"
        # so bracketing contracts rather than propagating a NaN.
        isfinite(ϕ) || return HZPoint(T(α), T(Inf), T(Inf))
        return HZPoint(T(α), T(ϕ), T(dϕ))
    end

    invalidate!(ev)
    ϕ_0, dϕ_0 = merit_ϕdϕ_at_zero(ev, u, du, ϕ0, dϕ0)
    isfinite(ϕ_0) || return LineSearchSolution(zero(T), ReturnCode.Failure)
    dϕ_0 ≥ zero(T) && return LineSearchSolution(zero(T), ReturnCode.Failure)

    θ = T(cache.θ)
    p = HZParams{T}(T(ϕ_0), T(dϕ_0), T(cache.ε) * abs(T(ϕ_0)), T(cache.δ), T(cache.σ))
    zero_pt = HZPoint(zero(T), T(ϕ_0), T(dϕ_0))

    c0 = ϕdϕ(min(T(cache.α_init), T(cache.α_max)))
    hz_wolfe(c0, p) && return hz_solution(ev, u, du, c0, ReturnCode.Success)

    a, b, found, best, ok = hz_bracket(
        p, ϕdϕ, c0, zero_pt, θ, T(cache.ρ), T(cache.α_max), cache.maxiters
    )
    found && return hz_solution(ev, u, du, best, ReturnCode.Success)
    ok || return hz_solution(ev, u, du, hz_best_effort(p, a, b), ReturnCode.Failure)

    for _ in 1:(cache.maxiters)
        width = b.α - a.α
        (hz_degenerate(a, b) || !isfinite(width)) && break
        A, B, found, best = hz_secant2(p, ϕdϕ, a, b, θ, cache.maxiters)
        found && return hz_solution(ev, u, du, best, ReturnCode.Success)
        # L2: force a bisection when secant² failed to shrink the bracket.
        if B.α - A.α > T(0.66) * width
            c = ϕdϕ((A.α + B.α) / 2)
            hz_wolfe(c, p) && return hz_solution(ev, u, du, c, ReturnCode.Success)
            A, B, found, best = hz_update(p, ϕdϕ, A, B, c, θ, cache.maxiters)
            found && return hz_solution(ev, u, du, best, ReturnCode.Success)
        end
        a, b = A, B
    end
    return hz_solution(ev, u, du, hz_best_effort(p, a, b), ReturnCode.Failure)
end

# Leaves the evaluator's caches holding the accepted point, so the caller can
# reuse the iterate and gradient there instead of re-evaluating. The common path
# accepts the point just probed, so this normally costs nothing.
@inline function hz_solution(ev, u, du, pt::HZPoint, retcode)
    ensure_evaluated_at!(ev, u, du, pt.α)
    return LineSearchSolution(pt.α, retcode, pt.ϕ, pt.dϕ)
end

function SciMLBase.reinit!(
        cache::HagerZhangLineSearchCache; p = missing, stats = missing, kwargs...
    )
    SciMLBase.reinit!(cache.merit_eval; p, stats)
    return cache
end

set_initial_step!(cache::HagerZhangLineSearchCache, α) = (cache.α_init = oftype(cache.α_init, α); cache)
