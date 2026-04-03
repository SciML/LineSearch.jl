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

@concrete mutable struct StrongWolfeLineSearchCache <: AbstractLineSearchCache
    ϕ
    dϕ
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

    ϕ = @closure (f, p, u, du, α, u_cache, fu_cache) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(f, fu_cache, u_cache, p)
        add_nf!(stats)
        return @fastmath norm(fu_cache)^2 / 2
    end

    dϕ = @closure (f, p, u, du, α, u_cache, fu_cache, deriv_op) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(f, fu_cache, u_cache, p)
        add_nf!(stats)
        return deriv_op(du, u_cache, fu_cache, p)
    end

    return StrongWolfeLineSearchCache(
        ϕ, dϕ, prob.f, prob.p, deriv_op, u_cache, fu_cache,
        T(alg.c1), T(alg.c2), T(alg.α_init), alg.maxiters, stats, alg
    )
end

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

@inline function _sw_phi(f, p, u, du, α)
    u_new = u .+ α .* du
    fu = f(u_new, p)
    @fastmath norm(fu)^2 / 2
end

@inline function _sw_dphi(grad_f, p, u, du, α)
    u_new = u .+ α .* du
    g = grad_f(u_new, p)
    dot(g, du)
end

@inline function _sw_zoom_static(f, grad_f, p, u, du,
        a_lo, a_hi, ϕ_0, dϕ_0, ϕ_lo, c1, c2, maxiters)
    T = typeof(a_lo)
    α_out = a_lo
    ok = false
    done = false

    for _ in 1:maxiters
        if !done
            ϕ_hi = _sw_phi(f, p, u, du, a_hi)
            dϕ_lo_val = _sw_dphi(grad_f, p, u, du, a_lo)
            dϕ_hi_val = _sw_dphi(grad_f, p, u, du, a_hi)

            α_j = _sw_interpolate(a_lo, a_hi, ϕ_lo, ϕ_hi, dϕ_lo_val, dϕ_hi_val)
            bracket = T(0.01) * abs(a_hi - a_lo)
            α_j = clamp(α_j,
                min(a_lo, a_hi) + bracket,
                max(a_lo, a_hi) - bracket)

            ϕ_j = _sw_phi(f, p, u, du, α_j)

            if (ϕ_j > ϕ_0 + c1 * α_j * dϕ_0) || (ϕ_j >= ϕ_lo)
                a_hi = α_j
            else
                dϕ_j = _sw_dphi(grad_f, p, u, du, α_j)
                if abs(dϕ_j) <= -c2 * dϕ_0
                    α_out, ok, done = α_j, true, true
                else
                    if dϕ_j * (a_hi - a_lo) >= zero(T)
                        a_hi = a_lo
                    end
                    a_lo = α_j
                    ϕ_lo = ϕ_j
                end
            end
        end
    end

    if !done
        α_out = a_lo
    end
    (α_out, ok)
end

@inline function _sw_search_static(f, grad_f, p, u, du, ϕ_0, dϕ_0,
        c1, c2, α_init, maxiters)
    T = typeof(α_init)
    α_out = zero(T)
    ok = false

    dϕ_0 >= zero(T) && return (zero(T), false)

    α_prev = zero(T)
    α_i = α_init
    ϕ_prev = ϕ_0
    done = false

    for i in 1:maxiters
        if !done
            ϕ_i = _sw_phi(f, p, u, du, α_i)

            if (ϕ_i > ϕ_0 + c1 * α_i * dϕ_0) || (ϕ_i >= ϕ_prev && i > 1)
                α_z, ok_z = _sw_zoom_static(f, grad_f, p, u, du,
                    α_prev, α_i, ϕ_0, dϕ_0, ϕ_prev, c1, c2, maxiters)
                α_out, ok, done = α_z, ok_z, true
            else
                dϕ_i = _sw_dphi(grad_f, p, u, du, α_i)

                if abs(dϕ_i) <= -c2 * dϕ_0
                    α_out, ok, done = α_i, true, true
                elseif dϕ_i >= zero(T)
                    α_z, ok_z = _sw_zoom_static(f, grad_f, p, u, du,
                        α_i, α_prev, ϕ_0, dϕ_0, ϕ_i, c1, c2, maxiters)
                    α_out, ok, done = α_z, ok_z, true
                else
                    α_prev = α_i
                    ϕ_prev = ϕ_i
                    α_i *= T(2)
                end
            end
        end
    end

    if !done
        α_out, ok = α_prev, true
    end
    (α_out, ok)
end

@inline function _sw_zoom(ϕ, dϕ, a_lo, a_hi, ϕ_0, dϕ_0, ϕ_lo, c1, c2, maxiters)
    T = typeof(a_lo)
    α_out = a_lo
    ok = false
    done = false

    for _ in 1:maxiters
        if !done
            ϕ_hi = ϕ(a_hi)
            dϕ_lo_val = dϕ(a_lo)
            dϕ_hi_val = dϕ(a_hi)

            α_j = _sw_interpolate(a_lo, a_hi, ϕ_lo, ϕ_hi, dϕ_lo_val, dϕ_hi_val)
            bracket = T(0.01) * abs(a_hi - a_lo)
            α_j = clamp(α_j,
                min(a_lo, a_hi) + bracket,
                max(a_lo, a_hi) - bracket)

            ϕ_j = ϕ(α_j)

            if (ϕ_j > ϕ_0 + c1 * α_j * dϕ_0) || (ϕ_j >= ϕ_lo)
                a_hi = α_j
            else
                dϕ_j = dϕ(α_j)
                if abs(dϕ_j) <= -c2 * dϕ_0
                    α_out, ok, done = α_j, true, true
                else
                    if dϕ_j * (a_hi - a_lo) >= zero(T)
                        a_hi = a_lo
                    end
                    a_lo = α_j
                    ϕ_lo = ϕ_j
                end
            end
        end
    end

    if !done
        α_out = a_lo
    end
    (α_out, ok)
end

@inline function _sw_search(ϕ, dϕ, ϕ_0, dϕ_0, c1, c2, α_init, maxiters)
    T = typeof(α_init)
    α_out = zero(T)
    ok = false

    dϕ_0 >= zero(T) && return (zero(T), false)

    α_prev = zero(T)
    α_i = α_init
    ϕ_prev = ϕ_0
    done = false

    for i in 1:maxiters
        if !done
            ϕ_i = ϕ(α_i)

            if (ϕ_i > ϕ_0 + c1 * α_i * dϕ_0) || (ϕ_i >= ϕ_prev && i > 1)
                α_z, ok_z = _sw_zoom(ϕ, dϕ, α_prev, α_i,
                    ϕ_0, dϕ_0, ϕ_prev, c1, c2, maxiters)
                α_out, ok, done = α_z, ok_z, true
            else
                dϕ_i = dϕ(α_i)

                if abs(dϕ_i) <= -c2 * dϕ_0
                    α_out, ok, done = α_i, true, true
                elseif dϕ_i >= zero(T)
                    α_z, ok_z = _sw_zoom(ϕ, dϕ, α_i, α_prev,
                        ϕ_0, dϕ_0, ϕ_i, c1, c2, maxiters)
                    α_out, ok, done = α_z, ok_z, true
                else
                    α_prev = α_i
                    ϕ_prev = ϕ_i
                    α_i *= T(2)
                end
            end
        end
    end

    if !done
        α_out, ok = α_prev, true
    end
    (α_out, ok)
end

function CommonSolve.solve!(cache::StrongWolfeLineSearchCache, u, du)
    T = promote_type(eltype(du), eltype(u))

    ϕ = @closure α -> cache.ϕ(cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache)
    dϕ = @closure α -> cache.dϕ(cache.f, cache.p, u, du, α,
        cache.u_cache, cache.fu_cache, cache.deriv_op)

    ϕ_0 = ϕ(zero(T))
    dϕ_0 = dϕ(zero(T))

    isfinite(ϕ_0) || return LineSearchSolution(cache.α, ReturnCode.Failure)
    dϕ_0 >= zero(T) && return LineSearchSolution(cache.α, ReturnCode.Failure)

    α, ok = _sw_search(ϕ, dϕ, ϕ_0, dϕ_0, cache.c1, cache.c2, cache.α, cache.maxiters)

    ok && return LineSearchSolution(α, ReturnCode.Success)
    return LineSearchSolution(cache.α, ReturnCode.Failure)
end

function CommonSolve.solve!(cache::StaticStrongWolfeLineSearchCache, u, du)
    T = promote_type(eltype(du), eltype(u))

    ϕ_0 = _sw_phi(cache.f, cache.p, u, du, zero(T))
    dϕ_0 = _sw_dphi(cache.grad_f, cache.p, u, du, zero(T))

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