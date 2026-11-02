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

The line search can be traced by `Reactant.@jit`/`@compile`, including inside a traced
solver loop that carries its cache. The compiled search follows the same acceptance rule
and step-length updates as the host one, but it evaluates both trial points of each
iteration, does not count function evaluations in `stats`, and returns a
`LineSearch.TracedLineSearchSolution` with traced `step_size` and `retcode`. Its
floating-point results can differ from the host's in the last bits, which can change the
accepted step when a trial merit lies at the acceptance threshold.

# Examples

```julia
using LineSearch

alg = RobustNonMonotoneLineSearch(M = 10, gamma = 1e-4, maxiters = 100)
```
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
    nsteps
    η_strategy
    n_exp::Int
    stats <: Union{SciMLBase.NLStats, Nothing}
    alg <: RobustNonMonotoneLineSearch
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::RobustNonMonotoneLineSearch, fu, u;
        stats::Union{SciMLBase.NLStats, Nothing} = nothing, kwargs...
    )
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    T = promote_type(eltype(fu), eltype(u))
    if ReactantCore.within_compile()
        # `@bb similar` returns a traced array itself, and a traced solver loop carrying
        # this cache rejects that aliasing.
        u_cache, fu_cache = u .+ zero(eltype(u)), fu .+ zero(eltype(fu))
    end

    ϕ = @closure (
        f, p, u, du, α, u_cache,
        fu_cache,
    ) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(f, fu_cache, u_cache, p)
        add_nf!(stats)
        return @fastmath norm(fu_cache)^alg.n_exp
    end

    fn₁ = merit_norm(fu)^alg.n_exp
    η_strategy = @closure (n, xₙ, fₙ) -> alg.η_strategy(fn₁, n, xₙ, fₙ)

    return RobustNonMonotoneLineSearchCache(
        prob.f, prob.p, ϕ, u_cache, fu_cache, T(1), alg.maxiters, fill(fn₁, alg.M),
        T(alg.gamma), T(alg.sigma_1), alg.M, T(alg.tau_min), T(alg.tau_max),
        maybe_traced(0), η_strategy, alg.n_exp, stats, alg
    )
end

function CommonSolve.solve!(cache::RobustNonMonotoneLineSearchCache, u, du)
    T = promote_type(eltype(du), eltype(u))
    ReactantCore.within_compile() && return solve_traced(cache, u, du, T)
    ϕ = @closure α -> cache.ϕ(cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache)

    f_norm_old = ϕ(zero(T))

    α₊, α₋ = T(cache.σ₁), T(cache.σ₁)
    η = cache.η_strategy(cache.nsteps, u, f_norm_old)
    f_bar = maximum(cache.history)

    for _ in 1:(cache.maxiters)
        f_norm = ϕ(α₊)
        f_norm ≤ f_bar + η - cache.γ * α₊ * f_norm_old && return LineSearchSolution(
            α₊, ReturnCode.Success
        )

        α₊ *= clamp(
            α₊ * f_norm_old / (f_norm + (T(2) * α₊ - T(1)) * f_norm_old),
            cache.τ_min,
            cache.τ_max
        )

        f_norm = ϕ(-α₋)
        f_norm ≤ f_bar + η - cache.γ * α₋ * f_norm_old && return LineSearchSolution(
            -α₋, ReturnCode.Success
        )

        α₋ *= clamp(
            α₋ * f_norm_old / (f_norm + (T(2) * α₋ - T(1)) * f_norm_old),
            cache.τ_min,
            cache.τ_max
        )
    end

    return LineSearchSolution(T(cache.σ₁), ReturnCode.Failure)
end

function callback_into_cache!(cache::RobustNonMonotoneLineSearchCache, fu)
    if ReactantCore.within_compile()
        # `mod(n - 1, M) + 1 == mod1(n, M)` without branching on the traced `n`
        slot = mod(cache.nsteps - 1, cache.M) + 1
        cache.history = ifelse.(
            (1:(cache.M)) .== slot, scaled_norm(fu)^cache.n_exp, cache.history
        )
        cache.nsteps += 1
        return
    end
    cache.history[mod1(cache.nsteps, cache.M)] = norm(fu)^cache.n_exp
    cache.nsteps += 1
    return
end

function SciMLBase.reinit!(
        cache::RobustNonMonotoneLineSearchCache; p = missing, stats = missing, kwargs...
    )
    p !== missing && (cache.p = p)
    stats !== missing && (cache.stats = stats)
    cache.σ₁ = oftype(cache.σ₁, cache.alg.sigma_1)
    cache.M = oftype(cache.M, cache.alg.M)
    cache.τ_min = oftype(cache.τ_min, cache.alg.tau_min)
    cache.τ_max = oftype(cache.τ_max, cache.alg.tau_max)
    cache.nsteps = maybe_traced(0)
    # NOTE: Don't zero out the stats here, since we don't own it
    return cache
end

# Both trial points are evaluated in every iteration; the step the host loop would return
# is then selected with `ifelse`. `stats` is not updated: the loop body is traced once.
function solve_traced(cache::RobustNonMonotoneLineSearchCache, u, du, T)
    (; f, p, n_exp, γ, τ_min, τ_max, maxiters) = cache
    fu_buf = cache.fu_cache
    f_norm_old = traced_merit(f, p, u, du, zero(T), fu_buf, n_exp)
    η = cache.η_strategy(cache.nsteps, u, f_norm_old)
    f_bar = maximum(cache.history)

    σ₁ = T(cache.σ₁)
    # `σ₁` is the traced scalar stored in the cache, and `@trace` writes its results back
    # into the objects it carries, so the carried step lengths must be fresh values.
    α₊ = σ₁ + zero(σ₁)
    α₋ = σ₁ + zero(σ₁)
    step = -α₋
    accepted = ReactantCore.promote_to_traced(false)
    iter = ReactantCore.promote_to_traced(0)

    ReactantCore.@trace track_numbers = false while (iter < maxiters) & !accepted
        f₊ = traced_merit(f, p, u, du, α₊, fu_buf, n_exp)
        accept₊ = f₊ ≤ f_bar + η - γ * α₊ * f_norm_old
        f₋ = traced_merit(f, p, u, du, -α₋, fu_buf, n_exp)
        accept₋ = f₋ ≤ f_bar + η - γ * α₋ * f_norm_old

        step = ifelse(accept₊, α₊, -α₋)
        accepted = accept₊ | accept₋

        r₊ = α₊ * f_norm_old / (f₊ + (2 * α₊ - 1) * f_norm_old)
        r₋ = α₋ * f_norm_old / (f₋ + (2 * α₋ - 1) * f_norm_old)
        # XLA's clamp maps NaN to a bound; Julia's clamp propagates it.
        α₊ = α₊ * ifelse(isnan(r₊), r₊, clamp(r₊, τ_min, τ_max))
        α₋ = α₋ * ifelse(isnan(r₋), r₋, clamp(r₋, τ_min, τ_max))
        iter = iter + 1
    end

    return TracedLineSearchSolution(
        ifelse(accepted, step, σ₁),
        ifelse(accepted, ReturnCode.Success, ReturnCode.Failure)
    )
end

function traced_merit(f, p, u, du, α, fu_buf, n_exp)
    # A fresh buffer per evaluation; out-of-place (and scalar) residuals need none.
    buf = SciMLBase.isinplace(f) ? similar(fu_buf) : fu_buf
    fu = evaluate_f!!(f, buf, u .+ α .* du, p)
    return scaled_norm(fu)^n_exp
end

# Reactant lowers `norm` to `sqrt(dot(x, x))`, which underflows and overflows for finite
# inputs, so the compiled path scales by the largest entry as `LinearAlgebra` does.
merit_norm(x) = ReactantCore.within_compile() ? scaled_norm(x) : norm(x)

function scaled_norm(x)
    s = maximum(abs, x)
    unscalable = iszero(s) | !isfinite(s)
    scaled = s * sqrt(sum(abs2, x ./ ifelse(unscalable, one(s), s)))
    return ifelse(unscalable, s, scaled)
end

"""
    TracedLineSearchSolution(step_size, retcode)

What `solve!` of [`RobustNonMonotoneLineSearch`](@ref) returns inside a Reactant
compilation: the fields of [`LineSearchSolution`](@ref) without `ϕ` and `dϕ`, with a
traced `step_size` and a traced `SciMLBase.ReturnCode`.
"""
struct TracedLineSearchSolution{S, R}
    step_size::S
    retcode::R
end
