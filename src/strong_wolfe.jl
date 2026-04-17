"""
    StrongWolfeLineSearch(; autodiff = nothing, c1 = 1e-4, c2 = 0.9,
        α_init = 1.0, α_max = 65536.0, maxiters::Int = 10,
        zoom_maxiters::Int = 10)

Strong Wolfe line search satisfying both Armijo (sufficient decrease) and
curvature conditions. Based on Nocedal & Wright, "Numerical Optimization" (2006),
Algorithms 3.5 and 3.6.

`autodiff` is the automatic differentiation backend to use for computing the
directional derivative. Must be specified if analytic jacobian/jvp/vjp is not
available.

`maxiters` bounds the outer bracketing loop (Alg. 3.5). `zoom_maxiters` bounds
the inner zoom loop (Alg. 3.6) independently.
"""
@kwdef @concrete struct StrongWolfeLineSearch <: AbstractLineSearchAlgorithm
    autodiff = nothing
    c1 = 1.0e-4
    c2 = 0.9
    α_init = 1.0
    α_max = 65536.0
    maxiters::Int = 10
    zoom_maxiters::Int = 10
end

@concrete mutable struct StrongWolfeLineSearchCache <: AbstractLineSearchCache
    f
    p
    deriv_op
    u_cache
    fu_cache
    c1
    c2
    α
    α_max
    maxiters::Int
    zoom_maxiters::Int
    stats <: Union{SciMLBase.NLStats, Nothing}
    alg <: StrongWolfeLineSearch
end

@concrete struct StaticStrongWolfeLineSearchCache <: AbstractLineSearchCache
    f
    grad_f
    p
    c1
    c2
    α_init
    α_max
    maxiters::Int
    zoom_maxiters::Int
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::StrongWolfeLineSearch,
        fu::Union{SArray, Number}, u::Union{SArray, Number};
        grad_f = nothing, kwargs...
    )
    grad_f === nothing && error(
        "StrongWolfeLineSearch requires `grad_f` for static (GPU) dispatch"
    )
    T = promote_type(eltype(fu), eltype(u))
    return StaticStrongWolfeLineSearchCache(
        prob.f, grad_f, prob.p,
        T(alg.c1), T(alg.c2), T(alg.α_init), T(alg.α_max),
        alg.maxiters, alg.zoom_maxiters
    )
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::StrongWolfeLineSearch, fu, u;
        autodiff = nothing, kwargs...
    )
    return generic_strongwolfe_init(prob, alg, fu, u; autodiff, kwargs...)
end

function generic_strongwolfe_init(
        prob::AbstractNonlinearProblem, alg::StrongWolfeLineSearch,
        fu, u; stats::Union{SciMLBase.NLStats, Nothing} = nothing,
        autodiff = nothing, kwargs...
    )
    autodiff = autodiff !== nothing ? autodiff : alg.autodiff
    _, _, deriv_op = construct_jvp_or_vjp_operator(prob, fu, u; autodiff)
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    T = promote_type(eltype(fu), eltype(u))
    return StrongWolfeLineSearchCache(
        prob.f, prob.p, deriv_op, u_cache, fu_cache,
        T(alg.c1), T(alg.c2), T(alg.α_init), T(alg.α_max),
        alg.maxiters, alg.zoom_maxiters, stats, alg
    )
end

@inline function _sw_interpolate(a_lo, a_hi, ϕ_lo, ϕ_hi, dϕ_lo, dϕ_hi)
    d1 = dϕ_lo + dϕ_hi - 3 * (ϕ_lo - ϕ_hi) / (a_lo - a_hi)
    desc = d1 * d1 - dϕ_lo * dϕ_hi
    d2 = sqrt(max(desc, zero(desc)))
    candidate = a_hi - (a_hi - a_lo) * ((dϕ_hi + d2 - d1) / (dϕ_hi - dϕ_lo + 2 * d2))
    candidate = ifelse(isfinite(candidate), candidate, (a_lo + a_hi) / 2)
    return ifelse(desc < 0, (a_lo + a_hi) / 2, candidate)
end

struct _SWNonlinearEval{F, G, P, U, D}
    f::F
    grad_f::G
    p::P
    u::U
    du::D
end

@inline function (e::_SWNonlinearEval)(α)
    u_new = e.u .+ α .* e.du
    fu = e.f(u_new, e.p)
    ϕ = sum(abs2, fu) / 2
    dϕ = dot(e.grad_f(u_new, e.p), e.du)
    return (ϕ, dϕ)
end

# N&W Algorithm 3.6. Fixed iteration count with `!done` guard is intentional
# for GPU warp-uniform execution; do not rewrite as early `break`.
@inline function _sw_zoom(
        eval_fn, a_lo, a_hi, ϕ_0, dϕ_0,
        ϕ_lo, dϕ_lo, ϕ_hi, dϕ_hi, c1, c2, maxiters
    )
    T = typeof(a_lo)
    α_out = a_lo
    ok = false
    done = false

    for _ in 1:maxiters
        if !done
            α_j = _sw_interpolate(a_lo, a_hi, ϕ_lo, ϕ_hi, dϕ_lo, dϕ_hi)
            bracket = T(0.01) * abs(a_hi - a_lo)
            α_j = clamp(
                α_j,
                min(a_lo, a_hi) + bracket,
                max(a_lo, a_hi) - bracket
            )
            ϕ_j, dϕ_j = eval_fn(α_j)

            if (ϕ_j > ϕ_0 + c1 * α_j * dϕ_0) || (ϕ_j >= ϕ_lo)
                a_hi = α_j
                ϕ_hi = ϕ_j
                dϕ_hi = dϕ_j
            else
                if abs(dϕ_j) <= -c2 * dϕ_0
                    α_out, ok, done = α_j, true, true
                else
                    if dϕ_j * (a_hi - a_lo) >= zero(T)
                        a_hi = a_lo
                        ϕ_hi = ϕ_lo
                        dϕ_hi = dϕ_lo
                    end
                    a_lo = α_j
                    ϕ_lo = ϕ_j
                    dϕ_lo = dϕ_j
                end
            end
        end
    end
    if !done
        α_out = a_lo
    end
    return (α_out, ok)
end

# N&W Algorithm 3.5.
@inline function _sw_search(eval_fn, ϕ_0, dϕ_0, c1, c2, α_init, α_max, maxiters, zoom_maxiters)
    T = typeof(α_init)

    dϕ_0 >= zero(T) && return (zero(T), false)

    α_prev = zero(T)
    α_i = α_init
    ϕ_prev = ϕ_0
    dϕ_prev = dϕ_0
    done = false
    α_out = zero(T)
    ok = false

    for i in 1:maxiters
        if !done
            ϕ_i, dϕ_i = eval_fn(α_i)

            if (ϕ_i > ϕ_0 + c1 * α_i * dϕ_0) || (ϕ_i >= ϕ_prev && i > 1)
                α_z, ok_z = _sw_zoom(
                    eval_fn, α_prev, α_i, ϕ_0, dϕ_0,
                    ϕ_prev, dϕ_prev, ϕ_i, dϕ_i, c1, c2, zoom_maxiters
                )
                α_out, ok, done = α_z, ok_z, true
            elseif abs(dϕ_i) <= -c2 * dϕ_0
                α_out, ok, done = α_i, true, true
            elseif dϕ_i >= zero(T)
                α_z, ok_z = _sw_zoom(
                    eval_fn, α_i, α_prev, ϕ_0, dϕ_0,
                    ϕ_i, dϕ_i, ϕ_prev, dϕ_prev, c1, c2, zoom_maxiters
                )
                α_out, ok, done = α_z, ok_z, true
            else
                α_prev = α_i
                ϕ_prev = ϕ_i
                dϕ_prev = dϕ_i
                α_i = min(α_i * T(2), α_max)
            end
        end
    end
    if !done
        α_out = α_i
        ok = false
    end
    return (α_out, ok)
end

function CommonSolve.solve!(cache::StrongWolfeLineSearchCache, u, du)
    T = promote_type(eltype(du), eltype(u))

    ϕdϕ = @closure α -> begin
        @bb @. cache.u_cache = u + α * du
        cache.fu_cache = evaluate_f!!(cache.f, cache.fu_cache, cache.u_cache, cache.p)
        add_nf!(cache.stats)
        obj = sum(abs2, cache.fu_cache) / 2
        deriv = cache.deriv_op(du, cache.u_cache, cache.fu_cache, cache.p)
        return obj, deriv
    end

    ϕ_0, dϕ_0 = ϕdϕ(zero(T))
    isfinite(ϕ_0) || return LineSearchSolution(cache.α, ReturnCode.Failure)
    dϕ_0 >= zero(T) && return LineSearchSolution(cache.α, ReturnCode.Failure)

    α, ok = _sw_search(
        ϕdϕ, ϕ_0, dϕ_0, cache.c1, cache.c2,
        cache.α, cache.α_max, cache.maxiters, cache.zoom_maxiters
    )
    ok && return LineSearchSolution(α, ReturnCode.Success)
    return LineSearchSolution(cache.α, ReturnCode.Failure)
end

function CommonSolve.solve!(cache::StaticStrongWolfeLineSearchCache, u, du)
    T = promote_type(eltype(du), eltype(u))

    eval_fn = _SWNonlinearEval(cache.f, cache.grad_f, cache.p, u, du)

    ϕ_0, dϕ_0 = eval_fn(zero(T))
    isfinite(ϕ_0) || return LineSearchSolution(T(cache.α_init), ReturnCode.Failure)
    dϕ_0 >= zero(T) && return LineSearchSolution(T(cache.α_init), ReturnCode.Failure)

    α, ok = _sw_search(
        eval_fn, ϕ_0, dϕ_0, cache.c1, cache.c2,
        T(cache.α_init), T(cache.α_max), cache.maxiters, cache.zoom_maxiters
    )
    ok && return LineSearchSolution(α, ReturnCode.Success)
    return LineSearchSolution(T(cache.α_init), ReturnCode.Failure)
end

function SciMLBase.reinit!(
        cache::StrongWolfeLineSearchCache; p = missing, stats = missing, kwargs...
    )
    p !== missing && (cache.p = p)
    stats !== missing && (cache.stats = stats)
    cache.α = oftype(cache.α, cache.alg.α_init)
    cache.c1 = oftype(cache.c1, cache.alg.c1)
    cache.c2 = oftype(cache.c2, cache.alg.c2)
    cache.α_max = oftype(cache.α_max, cache.alg.α_max)
    return cache
end
