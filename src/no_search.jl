"""
    NoLineSearch(alpha)

Don't perform a line search. Just return the initial step length of `alpha`.
"""
@kwdef @concrete struct NoLineSearch <: AbstractLineSearchAlgorithm
    alpha = true
end

@concrete mutable struct NoLineSearchCache <: AbstractLineSearchCache
    alpha
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::NoLineSearch, fu, u; kwargs...)
    return NoLineSearchCache(promote_type(eltype(fu), eltype(u))(alg.alpha))
end

function CommonSolve.solve!(cache::NoLineSearchCache, u, du)
    return LineSearchSolution(cache.alpha, ReturnCode.Success)
end
