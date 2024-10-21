"""
    LineSearchesJL(; method = LineSearches.Static(), autodiff = nothing,
        initial_alpha = true)

Wrapper over algorithms from
[LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl/). Allows automatic
construction of the objective functions for the line search algorithms utilizing automatic
differentiation for fast VJPs or JVPs.

!!! warning

    Needs `LineSearches.jl` to be explicitly loaded before using this functionality.

### Arguments

  - `method`: the line search algorithm to use. Defaults to
    `method = LineSearches.Static()`, which means that the step size is fixed to the value
    of `alpha`.
  - `autodiff`: the automatic differentiation backend to use for the line search. Must be
    specified if analytic jacobian/jvp/vjp is not available.
  - `initial_alpha`: the initial step size to use. Defaults to `true` (which is equivalent
    to `1`).
"""
struct LineSearchesJL{M, A, AD <: Union{Nothing, ADTypes.AbstractADType}} <:
       AbstractLineSearchAlgorithm
    method::M
    initial_alpha::A
    autodiff::AD

    function LineSearchesJL(method, initial_alpha, autodiff)
        if Base.get_extension(@__MODULE__, :LineSearchLineSearchesExt) === nothing
            error("LineSearches.jl is not loaded. Please load the extension with \
                   `using LineSearches`")
        end
        return new{typeof(method), typeof(initial_alpha), typeof(autodiff)}(
            method, initial_alpha, autodiff)
    end
end

@concrete mutable struct LineSearchesJLCache <: AbstractLineSearchCache
    f
    p
    ϕ
    dϕ
    ϕdϕ
    method
    alpha
    initial_alpha
    deriv_op
    u_cache
    fu_cache
    stats <: Union{SciMLBase.NLStats, Nothing}
    alg <: LineSearchesJL
end

# Both forward and reverse AD can be used for line-search.
# We prefer forward AD for better performance, however, reverse AD is also supported if
# user explicitly requests it.
function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::LineSearchesJL, fu, u;
        stats::Union{SciMLBase.NLStats, Nothing} = nothing, autodiff = nothing, kwargs...)
    T = promote_type(eltype(fu), eltype(u))
    autodiff = autodiff !== nothing ? autodiff : alg.autodiff

    _, _, deriv_op = construct_jvp_or_vjp_operator(prob, fu, u; autodiff)

    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)

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

    ϕdϕ = @closure (f, p, u, du, α, u_cache, fu_cache, deriv_op) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(f, fu_cache, u_cache, p)
        add_nf!(stats)
        deriv = deriv_op(du, u_cache, fu_cache, p)
        obj = @fastmath norm(fu_cache)^2 / 2
        return obj, deriv
    end

    return LineSearchesJLCache(
        prob.f, prob.p, ϕ, dϕ, ϕdϕ, alg.method, T(alg.initial_alpha), T(alg.initial_alpha),
        deriv_op, u_cache, fu_cache, stats, alg)
end

function CommonSolve.solve!(cache::LineSearchesJLCache, u, du)
    ϕ = @closure α -> cache.ϕ(cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache)
    dϕ = @closure α -> cache.dϕ(
        cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache, cache.deriv_op)
    ϕdϕ = @closure α -> cache.ϕdϕ(
        cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache, cache.deriv_op)

    ϕ₀, dϕ₀ = ϕdϕ(zero(eltype(u)))

    # Here we should be resetting the search direction for some algorithms especially
    # if we start mixing in jacobian reuse and such
    dϕ₀ ≥ 0 && return LineSearchSolution(one(eltype(u)), ReturnCode.Failure)

    # We can technically reduce 1 axpy by reusing the returned value from cache.method
    # but it's not worth the extra complexity
    cache.alpha = first(cache.method(ϕ, dϕ, ϕdϕ, cache.alpha, ϕ₀, dϕ₀))
    return LineSearchSolution(cache.alpha, ReturnCode.Success)
end

function SciMLBase.reinit!(
        cache::LineSearchesJLCache; p = missing, stats = missing, kwargs...)
    p !== missing && (cache.p = p)
    stats !== missing && (cache.stats = stats)
    cache.alpha = cache.initial_alpha
    # NOTE: Don't zero out the stats here, since we don't own it
    return cache
end
