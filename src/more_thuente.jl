"""
    MoreThuenteLineSearch(; autodiff = nothing, ftol = 1e-4, gtol = 0.9,
        xtol = 1e-10, α_init = 1.0, α_min = 0.0, α_max = Inf, maxiters = 50)

Moré–Thuente line search satisfying the strong Wolfe conditions, following the
MINPACK-2 `dcsrch`/`dcstep` formulation (Moré & Thuente, *ACM TOMS* 20(3), 1994).

Uses safeguarded cubic/quadratic interpolation and, during its first stage,
minimizes the auxiliary function `ψ(α) = ϕ(α) - ϕ(0) - ftol α ϕ'(0)` rather than
`ϕ` itself, which is what lets it locate a strong-Wolfe point in very few
evaluations on well-scaled problems.

Because it enforces the *original* strong Wolfe conditions, it stalls once
`ϕ(α) - ϕ(0)` reaches roundoff; prefer [`HagerZhangLineSearch`](@ref) when
iterating to tight tolerances.

Works with both merits: pass a `NonlinearProblem` for `‖F‖²/2` or an
`OptimizationProblem` for the objective itself.

# Keyword Arguments

- `autodiff`: AD backend for the directional derivative under the residual
  merit; unused for the objective merit.
- `ftol`: sufficient-decrease coefficient.
- `gtol`: curvature coefficient for `|ϕ'(α)| ≤ gtol |ϕ'(0)|`.
- `xtol`: relative width at which a bracketed interval is deemed converged.
- `α_init`, `α_min`, `α_max`: initial step and bounds.
- `maxiters`: maximum function evaluations.

# Examples

```julia
using LineSearch

alg = MoreThuenteLineSearch(ftol = 1e-4, gtol = 0.1)
```
"""
@kwdef @concrete struct MoreThuenteLineSearch <: AbstractLineSearchAlgorithm
    autodiff = nothing
    ftol = 1.0e-4
    gtol = 0.9
    xtol = 1.0e-10
    α_init = 1.0
    α_min = 0.0
    α_max = Inf
    maxiters::Int = 50
end

@concrete mutable struct MoreThuenteLineSearchCache <: AbstractLineSearchCache
    merit_eval
    ftol
    gtol
    xtol
    α_init
    α_min
    α_max
    maxiters::Int
    alg <: MoreThuenteLineSearch
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::MoreThuenteLineSearch, fu, u;
        stats::Union{SciMLBase.NLStats, Nothing} = nothing, autodiff = nothing, kwargs...
    )
    T = promote_type(eltype(fu), eltype(u))
    autodiff = autodiff !== nothing ? autodiff : alg.autodiff
    ev = init_merit(prob, fu, u; autodiff, stats)
    return build_mt_cache(ev, alg, T)
end

function CommonSolve.init(
        prob::OptimizationProblem, alg::MoreThuenteLineSearch, u;
        stats::Union{SciMLBase.NLStats, Nothing} = nothing, kwargs...
    )
    ev = init_merit(prob, u; stats)
    return build_mt_cache(ev, alg, real(eltype(u)))
end

function CommonSolve.init(
        prob::OptimizationProblem, alg::MoreThuenteLineSearch, gu, u; kwargs...
    )
    return CommonSolve.init(prob, alg, u; kwargs...)
end

function build_mt_cache(ev, alg::MoreThuenteLineSearch, ::Type{T}) where {T}
    return MoreThuenteLineSearchCache(
        ev, T(alg.ftol), T(alg.gtol), T(alg.xtol), T(alg.α_init),
        T(alg.α_min), T(alg.α_max), alg.maxiters, alg
    )
end

# MINPACK-2 `dcstep`: safeguarded interpolation producing the next trial step and
# the updated interval of uncertainty. Returns the new state as a tuple.
function mt_step(
        stx::T, fx::T, dx::T, sty::T, fy::T, dy::T,
        stp::T, fp::T, dp::T, brackt::Bool, stpmin::T, stpmax::T
    ) where {T}
    p66 = T(0.66)
    sgnd = dp * sign(dx)

    if fp > fx
        # Higher function value: the minimum is bracketed.
        θ = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(θ), abs(dx), abs(dp))
        γ = s * sqrt(max(zero(T), (θ / s)^2 - (dx / s) * (dp / s)))
        stp < stx && (γ = -γ)
        p = (γ - dx) + θ
        q = ((γ - dx) + γ) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2) * (stp - stx)
        stpf = abs(stpc - stx) < abs(stpq - stx) ? stpc : stpc + (stpq - stpc) / 2
        brackt = true
    elseif sgnd < zero(T)
        # Lower function value, derivatives of opposite sign: minimum bracketed.
        θ = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(θ), abs(dx), abs(dp))
        γ = s * sqrt(max(zero(T), (θ / s)^2 - (dx / s) * (dp / s)))
        stp > stx && (γ = -γ)
        p = (γ - dp) + θ
        q = ((γ - dp) + γ) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        stpf = abs(stpc - stp) > abs(stpq - stp) ? stpc : stpq
        brackt = true
    elseif abs(dp) < abs(dx)
        # Lower function value, same-sign derivatives, decreasing magnitude.
        θ = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(θ), abs(dx), abs(dp))
        # γ = 0 exactly when the cubic does not tend to infinity along the step.
        γ = s * sqrt(max(zero(T), (θ / s)^2 - (dx / s) * (dp / s)))
        stp > stx && (γ = -γ)
        p = (γ - dp) + θ
        q = (γ + (dx - dp)) + γ
        r = p / q
        stpc = if r < zero(T) && γ != zero(T)
            stp + r * (stx - stp)
        elseif stp > stx
            stpmax
        else
            stpmin
        end
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if brackt
            stpf = abs(stpc - stp) < abs(stpq - stp) ? stpc : stpq
            stpf = stp > stx ? min(stp + p66 * (sty - stp), stpf) :
                max(stp + p66 * (sty - stp), stpf)
        else
            stpf = abs(stpc - stp) > abs(stpq - stp) ? stpc : stpq
            stpf = min(stpmax, max(stpmin, stpf))
        end
    else
        # Lower function value, same-sign derivatives, non-decreasing magnitude.
        if brackt
            θ = 3 * (fp - fy) / (sty - stp) + dy + dp
            s = max(abs(θ), abs(dy), abs(dp))
            γ = s * sqrt(max(zero(T), (θ / s)^2 - (dy / s) * (dp / s)))
            stp > sty && (γ = -γ)
            p = (γ - dp) + θ
            q = ((γ - dp) + γ) + dy
            r = p / q
            stpf = stp + r * (sty - stp)
        else
            stpf = stp > stx ? stpmax : stpmin
        end
    end

    # Update the interval of uncertainty.
    if fp > fx
        sty, fy, dy = stp, fp, dp
    else
        if sgnd < zero(T)
            sty, fy, dy = stx, fx, dx
        end
        stx, fx, dx = stp, fp, dp
    end

    return (stx, fx, dx, sty, fy, dy, stpf, brackt)
end


function CommonSolve.solve!(
        cache::MoreThuenteLineSearchCache, u, du; ϕ0 = nothing, dϕ0 = nothing
    )
    T = promote_type(eltype(du), eltype(u))
    ev = cache.merit_eval

    ftol = T(cache.ftol)
    gtol = T(cache.gtol)
    xtol = T(cache.xtol)
    α_min = T(cache.α_min)
    α_max = T(cache.α_max)
    xtrapl = T(1.1)
    xtrapu = T(4.0)
    p66 = T(0.66)

    invalidate!(ev)
    ϕ_0, dϕ_0 = merit_ϕdϕ_at_zero(ev, u, du, ϕ0, dϕ0)
    ϕ_0, dϕ_0 = T(ϕ_0), T(dϕ_0)
    isfinite(ϕ_0) || return LineSearchSolution(zero(T), ReturnCode.Failure)
    dϕ_0 ≥ zero(T) && return LineSearchSolution(zero(T), ReturnCode.Failure)

    brackt = false
    stage = 1
    gtest = ftol * dϕ_0
    width = α_max - α_min
    width1 = width / T(0.5)
    if !isfinite(width)
        width = T(1.0e10)
        width1 = 2 * width
    end

    stx, fx, gx = zero(T), ϕ_0, dϕ_0
    sty, fy, gy = zero(T), ϕ_0, dϕ_0
    α0 = clamp(T(cache.α_init), α_min, α_max)
    stmin, stmax = zero(T), α0 + xtrapu * α0
    stp = α0
    best_α, best_ϕ, best_dϕ = zero(T), ϕ_0, dϕ_0

    for _ in 1:(cache.maxiters)
        f, g = merit_ϕdϕ(ev, u, du, stp)
        f, g = T(f), T(g)

        if !isfinite(f) || !isfinite(g)
            # Nothing to interpolate against; contract hard.
            stp = brackt ? stx + (sty - stx) / 2 : stp / 2
            if stp ≤ α_min
                ensure_evaluated_at!(ev, u, du, best_α)
                return LineSearchSolution(best_α, ReturnCode.Failure, best_ϕ, best_dϕ)
            end
            continue
        end
        if f < best_ϕ
            best_α, best_ϕ, best_dϕ = stp, f, g
        end

        ftest = ϕ_0 + stp * gtest
        (stage == 1 && f ≤ ftest && g ≥ zero(T)) && (stage = 2)

        # Strong Wolfe: accept.
        if f ≤ ftest && abs(g) ≤ gtol * (-dϕ_0)
            # `f`/`g` came from the probe just made, so the caches already hold
            # the accepted point.
            return LineSearchSolution(stp, ReturnCode.Success, f, g)
        end
        if brackt && ((stp ≤ stmin || stp ≥ stmax) || (stmax - stmin) ≤ xtol * stmax)
            ensure_evaluated_at!(ev, u, du, best_α)
            return LineSearchSolution(best_α, ReturnCode.Failure, best_ϕ, best_dϕ)
        end

        if stage == 1 && f ≤ fx && f > ftest
            # Interpolate on the auxiliary function ψ rather than ϕ.
            fm = f - stp * gtest
            fxm = fx - stx * gtest
            fym = fy - sty * gtest
            gm = g - gtest
            gxm = gx - gtest
            gym = gy - gtest
            stx, fxm, gxm, sty, fym, gym, stp, brackt = mt_step(
                stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax
            )
            fx = fxm + stx * gtest
            fy = fym + sty * gtest
            gx = gxm + gtest
            gy = gym + gtest
        else
            stx, fx, gx, sty, fy, gy, stp, brackt = mt_step(
                stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax
            )
        end

        if brackt
            abs(sty - stx) ≥ p66 * width1 && (stp = stx + (sty - stx) / 2)
            width1 = width
            width = abs(sty - stx)
            stmin = min(stx, sty)
            stmax = max(stx, sty)
        else
            stmin = stp + xtrapl * (stp - stx)
            stmax = stp + xtrapu * (stp - stx)
        end

        stp = clamp(stp, α_min, α_max)
        if (brackt && (stp ≤ stmin || stp ≥ stmax)) ||
                (brackt && (stmax - stmin) ≤ xtol * stmax)
            stp = stx
        end
        if stp ≤ zero(T)
            ensure_evaluated_at!(ev, u, du, best_α)
            return LineSearchSolution(best_α, ReturnCode.Failure, best_ϕ, best_dϕ)
        end
    end
    ensure_evaluated_at!(ev, u, du, best_α)
    return LineSearchSolution(best_α, ReturnCode.Failure, best_ϕ, best_dϕ)
end

function SciMLBase.reinit!(
        cache::MoreThuenteLineSearchCache; p = missing, stats = missing, kwargs...
    )
    SciMLBase.reinit!(cache.merit_eval; p, stats)
    return cache
end

set_initial_step!(cache::MoreThuenteLineSearchCache, α) = (cache.α_init = oftype(cache.α_init, α); cache)
