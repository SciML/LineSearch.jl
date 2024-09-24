"""
    LiFukushimaLineSearch(; lambda_0 = 1, beta = 1 // 2, sigma_1 = 1 // 1000,
        sigma_2 = 1 // 1000, eta = 1 // 10, nan_maxiters::Int = 5, maxiters::Int = 100)

A derivative-free line search and global convergence of Broyden-like method for nonlinear
equations [li2000derivative](@cite).
"""
@kwdef @concrete struct LiFukushimaLineSearch <: AbstractLineSearchAlgorithm
    lambda_0 = 1
    beta = 1 // 2
    sigma_1 = 1 // 1000
    sigma_2 = 1 // 1000
    eta = 1 // 10
    rho = 9 // 10
    nan_maxiters::Int = 5
    maxiters::Int = 100
end

@concrete mutable struct LiFukushimaLineSearchCache <: AbstractLineSearchCache
    ϕ
    f
    p
    internalnorm
    u_cache
    fu_cache
    λ₀
    β
    σ₁
    σ₂
    η
    ρ
    α
    nan_maxiters::Int
    maxiters::Int
    stats <: Union{SciMLBase.NLStats, Nothing}
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::LiFukushimaLineSearch, fu, u;
        stats::Union{SciMLBase.NLStats, Nothing} = nothing, kwargs...)
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
        ϕ, prob.f, prob.p, T(1), u_cache, fu_cache, alg.lambda_0, alg.beta,
        alg.sigma_1, alg.sigma_2, alg.eta, alg.rho, T(1), alg.nan_maxiters,
        alg.maxiters, stats)
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

    if !isfinite(fxλp_norm)
        nan_converged = false
        for _ in 1:(cache.nan_maxiters)
            λ₁, λ₂ = λ₂, cache.β * λ₂
            fxλp_norm = ϕ(λ₂)
            nan_converged = isfinite(fxλp_norm)
            nan_converged && break
        end
        nan_converged || return LineSearchSolution(cache.α, ReturnCode.Failure)
    end

    for i in 1:(cache.maxiters)
        fxλp_norm = ϕ(λ₂)
        converged = fxλp_norm ≤ (1 + cache.η) * fx_norm - cache.σ₁ * λ₂^2 * du_norm^2
        converged && return LineSearchSolution(λ₂, ReturnCode.Success)
        λ₁, λ₂ = λ₂, cache.β * λ₂
    end

    return LineSearchSolution(cache.α, ReturnCode.Failure)
end
