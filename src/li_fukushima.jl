"""
    LiFukushimaLineSearch(; lambda_0 = 1, beta = 1 // 2, sigma_1 = 1 // 1000,
        sigma_2 = 1 // 1000, eta = 1 // 10, nan_maxiters::Int = 5, maxiters::Int = 100)

A derivative-free line search and global convergence of Broyden-like method for nonlinear
equations [li2000derivative](@cite).

!!! tip

    For static arrays and numbers if `nan_maxiters` is either `nothing` or `missing`,
    we provide a fully non-allocating implementation of the algorithm, that can be used
    inside GPU kernels. However, this particular version doesn't support `stats` and
    `reinit!` and those will be ignored. Additionally, we fix the initial alpha for the
    search to be `1`.
"""
@kwdef @concrete struct LiFukushimaLineSearch <: AbstractLineSearchAlgorithm
    lambda_0 = 1
    beta = 1 // 2
    sigma_1 = 1 // 1000
    sigma_2 = 1 // 1000
    eta = 1 // 10
    rho = 9 // 10
    nan_maxiters <: Union{Missing, Nothing, Int} = 5
    maxiters::Int = 100
end

@concrete mutable struct LiFukushimaLineSearchCache <: AbstractLineSearchCache
    ϕ
    f
    p
    u_cache
    fu_cache
    λ₀
    β
    σ₁
    σ₂
    η
    ρ
    α
    nan_maxiters <: Union{Missing, Nothing, Int}
    maxiters::Int
    stats <: Union{SciMLBase.NLStats, Nothing}
    alg <: LiFukushimaLineSearch
end

@concrete struct StaticLiFukushimaLineSearchCache <: AbstractLineSearchCache
    f
    p
    λ₀
    β
    σ₁
    σ₂
    η
    ρ
    maxiters::Int
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::LiFukushimaLineSearch,
        fu::Union{SArray, Number}, u::Union{SArray, Number};
        stats::Union{SciMLBase.NLStats, Nothing} = nothing, kwargs...)
    if (alg.nan_maxiters === nothing || alg.nan_maxiters === missing) && stats === nothing
        T = promote_type(eltype(fu), eltype(u))
        return StaticLiFukushimaLineSearchCache(prob.f, prob.p, T(alg.lambda_0),
            T(alg.beta), T(alg.sigma_1), T(alg.sigma_2), T(alg.eta), T(alg.rho),
            alg.maxiters)
    end
    return generic_lifukushima_init(prob, alg, fu, u; stats, kwargs...)
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::LiFukushimaLineSearch, fu, u; kwargs...)
    return generic_lifukushima_init(prob, alg, fu, u; kwargs...)
end

function generic_lifukushima_init(
        prob::AbstractNonlinearProblem, alg::LiFukushimaLineSearch,
        fu, u; stats::Union{SciMLBase.NLStats, Nothing} = nothing, kwargs...)
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    T = promote_type(eltype(fu), eltype(u))

    ϕ = @closure (f, p, u, du, α, u_cache, fu_cache) -> begin
        @bb @. u_cache = u + α * du
        fu_cache = evaluate_f!!(f, fu_cache, u_cache, p)
        add_nf!(stats)
        return @fastmath norm(fu_cache)
    end

    return LiFukushimaLineSearchCache(
        ϕ, prob.f, prob.p, u_cache, fu_cache, T(alg.lambda_0), T(alg.beta),
        T(alg.sigma_1), T(alg.sigma_2), T(alg.eta), T(alg.rho), T(1), alg.nan_maxiters,
        alg.maxiters, stats, alg)
end

function CommonSolve.solve!(cache::LiFukushimaLineSearchCache, u, du)
    T = promote_type(eltype(du), eltype(u))
    ϕ = @closure α -> cache.ϕ(cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache)

    fx_norm = ϕ(zero(T))

    # Non-Blocking exit if the norm is NaN or Inf
    isfinite(fx_norm) || return LineSearchSolution(cache.α, ReturnCode.Failure)

    # Early Terminate based on Eq. 2.7
    du_norm = norm(du)
    fxλ_norm = ϕ(cache.α)
    fxλ_norm ≤ cache.ρ * fx_norm - cache.σ₂ * du_norm^2 && return LineSearchSolution(
        cache.α, ReturnCode.Success)

    λ₂, λ₁ = cache.λ₀, cache.λ₀
    fxλp_norm = ϕ(λ₂)

    if !isfinite(fxλp_norm) && cache.nan_maxiters !== nothing &&
       cache.nan_maxiters !== missing
        nan_converged = false
        for _ in 1:(cache.nan_maxiters)
            λ₁, λ₂ = λ₂, cache.β * λ₂
            fxλp_norm = ϕ(λ₂)
            nan_converged = isfinite(fxλp_norm)
            nan_converged && break
        end
        nan_converged || return LineSearchSolution(cache.α, ReturnCode.Failure)
    end

    for _ in 1:(cache.maxiters)
        fxλp_norm = ϕ(λ₂)
        converged = fxλp_norm ≤ (1 + cache.η) * fx_norm - cache.σ₁ * λ₂^2 * du_norm^2
        converged && return LineSearchSolution(λ₂, ReturnCode.Success)
        λ₁, λ₂ = λ₂, cache.β * λ₂
    end

    return LineSearchSolution(cache.α, ReturnCode.Failure)
end

function CommonSolve.solve!(cache::StaticLiFukushimaLineSearchCache, u, du)
    T = promote_type(eltype(du), eltype(u))

    fx_norm = norm(cache.f(u, cache.p))
    du_norm = norm(du)
    fxλ_norm = norm(cache.f(u .+ du, cache.p))

    if fxλ_norm ≤ cache.ρ * fx_norm - cache.σ₂ * du_norm^2
        return LineSearchSolution(T(true), ReturnCode.Success)
    end

    λ₂, λ₁ = cache.λ₀, cache.λ₀

    for _ in 1:(cache.maxiters)
        fxλp_norm = norm(cache.f(u .+ λ₂ .* du, cache.p))
        converged = fxλp_norm ≤ (1 + cache.η) * fx_norm - cache.σ₁ * λ₂^2 * du_norm^2
        converged && return LineSearchSolution(λ₂, ReturnCode.Success)
        λ₁, λ₂ = λ₂, cache.β * λ₂
    end

    return LineSearchSolution(T(true), ReturnCode.Failure)
end

function SciMLBase.reinit!(
        cache::LiFukushimaLineSearchCache; p = missing, stats = missing, kwargs...)
    p !== missing && (cache.p = p)
    stats !== missing && (cache.stats = stats)
    cache.α = oftype(cache.α, true)
    cache.λ₀ = oftype(cache.λ₀, cache.alg.lambda_0)
    cache.β = oftype(cache.β, cache.alg.beta)
    cache.σ₁ = oftype(cache.σ₁, cache.alg.sigma_1)
    cache.σ₂ = oftype(cache.σ₂, cache.alg.sigma_2)
    cache.η = oftype(cache.η, cache.alg.eta)
    cache.ρ = oftype(cache.ρ, cache.alg.rho)
    # NOTE: Don't zero out the stats here, since we don't own it
    return cache
end
