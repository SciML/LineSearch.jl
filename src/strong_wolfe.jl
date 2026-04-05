"""
    StrongWolfeLineSearch(; autodiff = nothing, c1 = 1e-4, c2 = 0.9,
        α_init = 1.0, maxiters::Int = 10)

Strong Wolfe line search satisfying both Armijo (sufficient decrease) and
curvature conditions. Based on Nocedal & Wright, "Numerical Optimization" (2006),
Algorithms 3.5 and 3.6.

`autodiff` is the automatic differentiation backend to use for computing the
directional derivative. Must be specified if analytic jacobian/jvp/vjp is not
available.
"""
@kwdef @concrete struct StrongWolfeLineSearch <: AbstractLineSearchAlgorithm
    autodiff = nothing
    c1 = 1e-4
    c2 = 0.9
    α_init = 1.0
    maxiters::Int = 10
end

# CPU path: stores raw ingredients, closure built fresh in solve!
@concrete mutable struct StrongWolfeLineSearchCache <: AbstractLineSearchCache
    f
    p
    deriv_op
    u_cache
    fu_cache
    c1
    c2
    α
    maxiters::Int
    stats <: Union{SciMLBase.NLStats, Nothing}
    alg <: StrongWolfeLineSearch
end

# GPU path: static cache for SArray/Number, no allocations
@concrete struct StaticStrongWolfeLineSearchCache <: AbstractLineSearchCache
    f
    grad_f
    p
    c1
    c2
    α_init
    maxiters::Int
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::StrongWolfeLineSearch,
        fu::Union{SArray, Number}, u::Union{SArray, Number};
        stats::Union{SciMLBase.NLStats, Nothing} = nothing,
        grad_f = nothing, kwargs...
    )
    if stats === nothing
        grad_f === nothing && error(
            "StrongWolfeLineSearch requires `grad_f` for static (GPU) dispatch"
        )
        T = promote_type(eltype(fu), eltype(u))
        return StaticStrongWolfeLineSearchCache(
            prob.f, grad_f, prob.p, T(alg.c1), T(alg.c2), T(alg.α_init), alg.maxiters
        )
    end
    return generic_strongwolfe_init(prob, alg, fu, u; stats, kwargs...)
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
        T(alg.c1), T(alg.c2), T(alg.α_init), alg.maxiters, stats, alg
    )
end

# cubic interpolation with bisection fallback when discriminant is negative
@inline function _sw_interpolate(a_lo, a_hi, ϕ_lo, ϕ_hi, dϕ_lo, dϕ_hi)
    d1 = dϕ_lo + dϕ_hi - 3 * (ϕ_lo - ϕ_hi) / (a_lo - a_hi)
    desc = d1 * d1 - dϕ_lo * dϕ_hi
    d2 = sqrt(max(desc, zero(desc)))
    ifelse(
        desc < 0,
        (a_lo + a_hi) / 2,
        a_hi - (a_hi - a_lo) * ((dϕ_hi + d2 - d1) / (dϕ_hi - dϕ_lo + 2 * d2))
    )
end

# combined f + grad_f eval for static/GPU path
@inline function _sw_phi_dphi(f, grad_f, p, u, du, α)
    u_new = u .+ α .* du
    fu = f(u_new, p)
    ϕ = @fastmath norm(fu)^2 / 2
    g = grad_f(u_new, p)
    dϕ = dot(g, du)
    return ϕ, dϕ
end

# N&W Algorithm 3.6 (zoom) for static path
# uses `done` flag instead of early return for GPU kernel compatibility
@inline function _sw_zoom_static(f, grad_f, p, u, du,
        a_lo, a_hi, ϕ_0, dϕ_0, ϕ_lo, dϕ_lo, c1, c2, maxiters)
    T = typeof(a_lo)
    α_out = a_lo
    ok = false
    done = false

    # evaluate a_hi once, then maintain via updates below
    ϕ_hi, dϕ_hi = _sw_phi_dphi(f, grad_f, p, u, du, a_hi)

    for _ in 1:maxiters
        if !done
            α_j = _sw_interpolate(a_lo, a_hi, ϕ_lo, ϕ_hi, dϕ_lo, dϕ_hi)
            bracket = T(0.01) * abs(a_hi - a_lo)
            α_j = clamp(α_j,
                min(a_lo, a_hi) + bracket,
                max(a_lo, a_hi) - bracket)

            ϕ_j, dϕ_j = _sw_phi_dphi(f, grad_f, p, u, du, α_j)

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

# N&W Algorithm 3.5 (outer search) for static path
@inline function _sw_search_static(f, grad_f, p, u, du, ϕ_0, dϕ_0,
        c1, c2, α_init, maxiters)
    T = typeof(α_init)
    α_max = T(65536)

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
            ϕ_i, dϕ_i = _sw_phi_dphi(f, grad_f, p, u, du, α_i)

            if (ϕ_i > ϕ_0 + c1 * α_i * dϕ_0) || (ϕ_i >= ϕ_prev && i > 1)
                α_z, ok_z = _sw_zoom_static(f, grad_f, p, u, du,
                    α_prev, α_i, ϕ_0, dϕ_0, ϕ_prev, dϕ_prev,
                    c1, c2, maxiters)
                α_out, ok, done = α_z, ok_z, true
            elseif abs(dϕ_i) <= -c2 * dϕ_0
                α_out, ok, done = α_i, true, true
            elseif dϕ_i >= zero(T)
                α_z, ok_z = _sw_zoom_static(f, grad_f, p, u, du,
                    α_i, α_prev, ϕ_0, dϕ_0, ϕ_i, dϕ_i,
                    c1, c2, maxiters)
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
        α_out = α_prev
        ok = false
    end
    return (α_out, ok)
end

# N&W Algorithm 3.6 (zoom) for mutable path
@inline function _sw_zoom(ϕdϕ, a_lo, a_hi, ϕ_0, dϕ_0,
        ϕ_lo, dϕ_lo, c1, c2, maxiters)
    T = typeof(a_lo)
    α_out = a_lo
    ok = false
    done = false

    ϕ_hi, dϕ_hi = ϕdϕ(a_hi)

    for _ in 1:maxiters
        if !done
            α_j = _sw_interpolate(a_lo, a_hi, ϕ_lo, ϕ_hi, dϕ_lo, dϕ_hi)
            bracket = T(0.01) * abs(a_hi - a_lo)
            α_j = clamp(α_j,
                min(a_lo, a_hi) + bracket,
                max(a_lo, a_hi) - bracket)

            ϕ_j, dϕ_j = ϕdϕ(α_j)

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

# N&W Algorithm 3.5 (outer search) for mutable path
@inline function _sw_search(ϕdϕ, ϕ_0, dϕ_0, c1, c2, α_init, maxiters)
    T = typeof(α_init)
    α_max = T(65536)

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
            ϕ_i, dϕ_i = ϕdϕ(α_i)

            if (ϕ_i > ϕ_0 + c1 * α_i * dϕ_0) || (ϕ_i >= ϕ_prev && i > 1)
                α_z, ok_z = _sw_zoom(ϕdϕ, α_prev, α_i,
                    ϕ_0, dϕ_0, ϕ_prev, dϕ_prev, c1, c2, maxiters)
                α_out, ok, done = α_z, ok_z, true
            elseif abs(dϕ_i) <= -c2 * dϕ_0
                α_out, ok, done = α_i, true, true
            elseif dϕ_i >= zero(T)
                α_z, ok_z = _sw_zoom(ϕdϕ, α_i, α_prev,
                    ϕ_0, dϕ_0, ϕ_i, dϕ_i, c1, c2, maxiters)
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
        α_out = α_prev
        ok = false
    end
    return (α_out, ok)
end

# closure built here, not in init, so it always reads cache.stats
function CommonSolve.solve!(cache::StrongWolfeLineSearchCache, u, du)
    T = promote_type(eltype(du), eltype(u))

    ϕdϕ = @closure α -> begin
        @bb @. cache.u_cache = u + α * du
        cache.fu_cache = evaluate_f!!(cache.f, cache.fu_cache, cache.u_cache, cache.p)
        add_nf!(cache.stats)
        obj = @fastmath norm(cache.fu_cache)^2 / 2
        deriv = cache.deriv_op(du, cache.u_cache, cache.fu_cache, cache.p)
        return obj, deriv
    end

    ϕ_0, dϕ_0 = ϕdϕ(zero(T))

    isfinite(ϕ_0) || return LineSearchSolution(cache.α, ReturnCode.Failure)
    dϕ_0 >= zero(T) && return LineSearchSolution(cache.α, ReturnCode.Failure)

    α, ok = _sw_search(ϕdϕ, ϕ_0, dϕ_0, cache.c1, cache.c2, cache.α, cache.maxiters)

    ok && return LineSearchSolution(α, ReturnCode.Success)
    return LineSearchSolution(cache.α, ReturnCode.Failure)
end

function CommonSolve.solve!(cache::StaticStrongWolfeLineSearchCache, u, du)
    T = promote_type(eltype(du), eltype(u))

    ϕ_0, dϕ_0 = _sw_phi_dphi(cache.f, cache.grad_f, cache.p, u, du, zero(T))

    isfinite(ϕ_0) || return LineSearchSolution(T(cache.α_init), ReturnCode.Failure)
    dϕ_0 >= zero(T) && return LineSearchSolution(T(cache.α_init), ReturnCode.Failure)

    α, ok = _sw_search_static(cache.f, cache.grad_f, cache.p, u, du,
        ϕ_0, dϕ_0, cache.c1, cache.c2, T(cache.α_init), cache.maxiters)

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
    return cache
end