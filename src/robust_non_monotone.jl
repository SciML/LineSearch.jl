"""
    RobustNonMonotoneLineSearch(; gamma = 1 // 10000, sigma_1 = 1, M::Int = 10,
        tau_min = 1 // 10, tau_max = 1 // 2, n_exp::Int = 2, maxiters::Int = 100,
        η_strategy = (fn₁, n, uₙ, fₙ) -> fn₁ / n^2)

Robust NonMonotone Line Search is a derivative free line search method from DF Sane
[la2006spectral](@cite).

### Keyword Arguments

  - `M`: The monotonicity of the algorithm is determined by a this positive integer.
    A value of 1 for `M` would result in strict monotonicity in the decrease of the L2-norm
    of the function `f`. However, higher values allow for more flexibility in this reduction.
    Despite this, the algorithm still ensures global convergence through the use of a
    non-monotone line-search algorithm that adheres to the Grippo-Lampariello-Lucidi
    condition. Values in the range of 5 to 20 are usually sufficient, but some cases may
    call for a higher value of `M`. The default setting is 10.
  - `gamma`: a parameter that influences if a proposed step will be accepted. Higher value
    of `gamma` will make the algorithm more restrictive in accepting steps. Defaults to
    `1e-4`.
  - `tau_min`: if a step is rejected the new step size will get multiplied by factor, and
    this parameter is the minimum value of that factor. Defaults to `0.1`.
  - `tau_max`: if a step is rejected the new step size will get multiplied by factor, and
    this parameter is the maximum value of that factor. Defaults to `0.5`.
  - `n_exp`: the exponent of the loss, i.e. ``f_n=||F(x_n)||^{n\\_exp}``. The paper uses
    `n_exp ∈ {1, 2}`. Defaults to `2`.
  - `η_strategy`:  function to determine the parameter `η`, which enables growth
    of ``||f_n||^2``. Called as `η = η_strategy(fn_1, n, x_n, f_n)` with `fn_1` initialized
    as ``fn_1=||f(x_1)||^{n\\_exp}``, `n` is the iteration number, `x_n` is the current
    `x`-value and `f_n` the current residual. Should satisfy ``η > 0`` and ``∑ₖ ηₖ < ∞``.
    Defaults to ``fn_1 / n^2``.
  - `maxiters`: the maximum number of iterations allowed for the inner loop of the
    algorithm. Defaults to `100`.
"""
@kwdef @concrete struct RobustNonMonotoneLineSearch <: AbstractLineSearchAlgorithm
    gamma = 1 // 10000
    sigma_1 = 1
    M::Int = 10
    tau_min = 1 // 10
    tau_max = 1 // 2
    n_exp::Int = 2
    maxiters::Int = 100
    η_strategy = (fn₁, n, uₙ, fₙ) -> fn₁ / n^2
end

@concrete mutable struct RobustNonMonotoneLineSearchCache <: AbstractLineSearchCache
    f
    p
    ϕ
    u_cache
    fu_cache
    internalnorm
    maxiters::Int
    history
    γ
    σ₁
    M::Int
    τ_min
    τ_max
    nsteps::Int
    η_strategy
    n_exp::Int
    stats <: Union{SciMLBase.NLStats, Nothing}
    alg <: RobustNonMonotoneLineSearch
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::RobustNonMonotoneLineSearch, fu, u;
        stats::Union{SciMLBase.NLStats, Nothing} = nothing, kwargs...)
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    T = promote_type(eltype(fu), eltype(u))

    ϕ = @closure (f, p, u, du, α, u_cache,
        fu_cache) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(f, fu_cache, u_cache, p)
        add_nf!(stats)
        return @fastmath norm(fu_cache)^alg.n_exp
    end

    fn₁ = norm(fu)^alg.n_exp
    η_strategy = @closure (n, xₙ, fₙ) -> alg.η_strategy(fn₁, n, xₙ, fₙ)

    return RobustNonMonotoneLineSearchCache(
        prob.f, prob.p, ϕ, u_cache, fu_cache, T(1), alg.maxiters, fill(fn₁, alg.M),
        T(alg.gamma), T(alg.sigma_1), alg.M, T(alg.tau_min), T(alg.tau_max), 0, η_strategy,
        alg.n_exp, stats, alg)
end

function CommonSolve.solve!(cache::RobustNonMonotoneLineSearchCache, u, du)
    T = promote_type(eltype(du), eltype(u))
    ϕ = @closure α -> cache.ϕ(cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache)

    f_norm_old = ϕ(zero(T))

    α₊, α₋ = T(cache.σ₁), T(cache.σ₁)
    η = cache.η_strategy(cache.nsteps, u, f_norm_old)
    f_bar = maximum(cache.history)

    for _ in 1:(cache.maxiters)
        f_norm = ϕ(α₊)
        f_norm ≤ f_bar + η - cache.γ * α₊ * f_norm_old && return LineSearchSolution(
            α₊, ReturnCode.Success)

        α₊ *= clamp(
            α₊ * f_norm_old / (f_norm + (T(2) * α₊ - T(1)) * f_norm_old),
            cache.τ_min,
            cache.τ_max
        )

        f_norm = ϕ(-α₋)
        f_norm ≤ f_bar + η - cache.γ * α₋ * f_norm_old && return LineSearchSolution(
            -α₋, ReturnCode.Success)

        α₋ *= clamp(
            α₋ * f_norm_old / (f_norm + (T(2) * α₋ - T(1)) * f_norm_old),
            cache.τ_min,
            cache.τ_max
        )
    end

    return LineSearchSolution(T(cache.σ₁), ReturnCode.Failure)
end

function callback_into_cache!(cache::RobustNonMonotoneLineSearchCache, fu)
    cache.history[mod1(cache.nsteps, cache.M)] = norm(fu)^cache.n_exp
    cache.nsteps += 1
    return
end

function SciMLBase.reinit!(
        cache::RobustNonMonotoneLineSearchCache; p = missing, stats = missing, kwargs...)
    p !== missing && (cache.p = p)
    stats !== missing && (cache.stats = stats)
    cache.σ₁ = oftype(cache.σ₁, cache.alg.sigma_1)
    cache.M = oftype(cache.M, cache.alg.M)
    cache.τ_min = oftype(cache.τ_min, cache.alg.tau_min)
    cache.τ_max = oftype(cache.τ_max, cache.alg.tau_max)
    cache.nsteps = 0
    # NOTE: Don't zero out the stats here, since we don't own it
    return cache
end
