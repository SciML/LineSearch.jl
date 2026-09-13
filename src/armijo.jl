"""
    ArmijoLineSearch(; autodiff = nothing, c_1 = 1e-4, contraction = 0.5,
        initial_alpha = 1, maxiters = 40)

Geometric Armijo backtracking [armijo1966minimization](@cite). Starting from
`initial_alpha`, multiply the step by `contraction` until the finite merit value
satisfies `ϕ(α) ≤ ϕ(0) + c_1 * α * ϕ'(0)`, for at most `maxiters` trials.
The starting directional derivative must be finite and strictly negative.
Unlike [`BackTracking`](@ref), this method does not interpolate trial values.
It enforces sufficient decrease only, so methods requiring a Wolfe curvature
condition should select a Wolfe search instead.

Uses the same `init` signatures as [`BackTracking`](@ref). `autodiff` supplies the
derivative backend for nonlinear problems without analytic derivatives. Call
`solve!(cache, u, du; ϕ0 = nothing, dϕ0 = nothing, gradient = nothing)` to reuse an
existing starting merit and directional derivative (or merit gradient).
If the caller always supplies the derivative, pass `need_deriv = false` to `init`
to avoid constructing a derivative operator. This also permits objective-only
`OptimizationFunction`s.

[`set_initial_step!`](@ref) controls the next search's first step, and
[`get_trial`](@ref) returns its cached point and residual after success.
For box projection, use [`ProjectedBackTracking`](@ref).
"""
@concrete struct ArmijoLineSearch <: AbstractLineSearchAlgorithm
    autodiff
    c_1
    contraction
    initial_alpha
    maxiters::Int
end

function ArmijoLineSearch(;
        autodiff = nothing, c_1 = 1.0e-4, contraction = 0.5,
        initial_alpha = 1, maxiters::Int = 40
    )
    validate_armijo(c_1, contraction, initial_alpha, maxiters)
    return ArmijoLineSearch(autodiff, c_1, contraction, initial_alpha, maxiters)
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::ArmijoLineSearch, fu, u;
        stats = nothing, autodiff = nothing, need_deriv = true, kwargs...
    )
    autodiff = autodiff === nothing ? alg.autodiff : autodiff
    ev = init_merit(prob, fu, u; stats, autodiff, need_deriv)
    return build_armijo_cache(ev, alg, u, nothing, nothing)
end

function CommonSolve.init(
        prob::OptimizationProblem, alg::ArmijoLineSearch, u;
        stats = nothing, need_deriv = true, kwargs...
    )
    ev = init_merit(prob, u; stats, need_deriv)
    return build_armijo_cache(ev, alg, u, nothing, nothing)
end

function CommonSolve.init(prob::OptimizationProblem, alg::ArmijoLineSearch, gu, u; kwargs...)
    return CommonSolve.init(prob, alg, u; kwargs...)
end

"""
    ProjectedBackTracking(; c_1 = 1e-4, contraction = 0.5,
        initial_alpha = 1, maxiters = 40)

Armijo backtracking along the box-projected path `P(u + α * du)`. Initialize with
`init(prob, alg, fu, u; lb = prob.lb, ub = prob.ub)` for a nonlinear problem, or
`init(prob, alg, u; lb = prob.lb, ub = prob.ub)` for an optimization problem.
Bounds may be scalars, arrays with the same axes as `u`, or `nothing` (unbounded).
The starting point must be feasible and the state must be real floating point.

Call `solve!(cache, u, du; gradient, ϕ0 = nothing)` with the merit gradient at `u`:
`J' * fu` for residual merit, or the objective gradient for optimization. Supplying
`ϕ0` avoids evaluating the starting merit. No derivative operator is constructed.
A trial is accepted when its finite merit satisfies
`ϕ(trial) ≤ ϕ0 + c_1 * dot(gradient, trial - u)` with a strictly negative slope.
Each rejection multiplies `α` by `contraction`, for at most `maxiters` trials.

Use [`get_trial`](@ref) after a successful solve to retrieve the projected point
and cached residual. Applying `u + sol.step_size * du` alone does not project the
step. Cache storage is reused across searches; `reinit!(cache; p)` updates parameters
and resets the initial step, and [`set_initial_step!`](@ref) changes the next search's
initial step. Bounds are fixed for the lifetime of the cache.

This is the Armijo rule along the projection arc [bertsekas1976goldstein](@cite).
With a general supplied direction it is a sufficient-decrease search; convergence
also depends on the outer method producing suitable descent directions.
"""
@concrete struct ProjectedBackTracking <: AbstractLineSearchAlgorithm
    c_1
    contraction
    initial_alpha
    maxiters::Int
end

function ProjectedBackTracking(;
        c_1 = 1.0e-4, contraction = 0.5, initial_alpha = 1, maxiters::Int = 40
    )
    validate_armijo(c_1, contraction, initial_alpha, maxiters)
    return ProjectedBackTracking(c_1, contraction, initial_alpha, maxiters)
end

@concrete mutable struct ArmijoCache <: AbstractLineSearchCache
    merit_eval
    step
    lb
    ub
    alpha
    alg <: Union{ArmijoLineSearch, ProjectedBackTracking}
end

function CommonSolve.init(
        prob::AbstractNonlinearProblem, alg::ProjectedBackTracking, fu, u;
        lb = prob.lb, ub = prob.ub, stats = nothing, kwargs...
    )
    ev = init_merit(prob, fu, u; stats, need_deriv = false)
    return build_armijo_cache(ev, alg, u, lb, ub)
end

function CommonSolve.init(
        prob::OptimizationProblem, alg::ProjectedBackTracking, u;
        lb = prob.lb, ub = prob.ub, stats = nothing, kwargs...
    )
    ev = init_merit(prob, u; stats, need_deriv = false)
    return build_armijo_cache(ev, alg, u, lb, ub)
end

function CommonSolve.init(prob::OptimizationProblem, alg::ProjectedBackTracking, gu, u; kwargs...)
    return CommonSolve.init(prob, alg, u; kwargs...)
end

function build_armijo_cache(ev, alg, u, lb, ub)
    T = eltype(u)
    T <: AbstractFloat || throw(ArgumentError("Armijo searches require real floating-point states."))
    lb = lb === nothing ? T(-Inf) : lb
    ub = ub === nothing ? T(Inf) : ub
    for bound in (lb, ub)
        bound isa Number || axes(bound) == axes(u) || throw(DimensionMismatch("Bounds must have the same axes as u."))
    end
    all(lb .<= ub) || throw(ArgumentError("Lower bounds must not exceed upper bounds."))
    all(isfinite, u) && all(lb .<= u .<= ub) || throw(ArgumentError("The starting point must be finite and feasible."))
    @bb step = similar(u)
    return ArmijoCache(ev, step, copy(lb), copy(ub), T(alg.initial_alpha), alg)
end

function CommonSolve.solve!(cache::ArmijoCache, u, du; gradient = nothing, ϕ0 = nothing, dϕ0 = nothing)
    ev = cache.merit_eval
    all(isfinite, u) && all(cache.lb .<= u .<= cache.ub) || throw(ArgumentError("The starting point must be finite and feasible."))
    invalidate!(ev)
    cost, derivative = armijo_initial(cache.alg, ev, u, du, gradient, ϕ0, dϕ0)
    alpha = cache.alpha
    for _ in 1:cache.alg.maxiters
        trial, step = ev.u_cache, cache.step
        @bb @. trial = clamp(u + alpha * du, cache.lb, cache.ub)
        @bb @. step = trial - u
        ev.u_cache, cache.step = trial, step
        slope = armijo_slope(cache.alg, gradient, step, derivative, alpha)
        if isfinite(slope) && slope < 0
            value = merit_value!(ev.merit, ev, trial)
            if isfinite(value) && value <= cost + cache.alg.c_1 * slope
                return LineSearchSolution(alpha, ReturnCode.Success, value, nothing)
            end
        end
        alpha *= oftype(alpha, cache.alg.contraction)
    end
    return LineSearchSolution(zero(alpha), ReturnCode.Failure)
end

"""
    get_trial(cache)

Return `(u = trial_point, fu = trial_residual)` from the most recent successful
[`ArmijoLineSearch`](@ref) or [`ProjectedBackTracking`](@ref) solve. `fu` is `nothing` for optimization problems.
The returned arrays alias cache storage and may be overwritten by the next search;
copy them if they must remain available. The result is unspecified after a failed search.
"""
function get_trial(cache::ArmijoCache)
    ev = cache.merit_eval
    return (; u = ev.u_cache, fu = ev.merit isa ResidualMerit ? ev.fu_cache : nothing)
end

function SciMLBase.reinit!(cache::ArmijoCache; kwargs...)
    SciMLBase.reinit!(cache.merit_eval; kwargs...)
    cache.alpha = oftype(cache.alpha, cache.alg.initial_alpha)
    return cache
end

function set_initial_step!(cache::ArmijoCache, alpha)
    isfinite(alpha) && alpha > 0 || throw(ArgumentError("The initial step must be positive and finite."))
    cache.alpha = oftype(cache.alpha, alpha)
    return cache
end

function armijo_initial(::ProjectedBackTracking, ev, u, du, gradient, value, derivative)
    gradient === nothing && throw(ArgumentError("ProjectedBackTracking requires the merit gradient at u."))
    return (value === nothing ? merit_value!(ev.merit, ev, u) : value), nothing
end

armijo_slope(::ProjectedBackTracking, gradient, step, derivative, alpha) = real(dot(gradient, step))
armijo_slope(::ArmijoLineSearch, gradient, step, derivative, alpha) = alpha * derivative

function armijo_initial(::ArmijoLineSearch, ev, u, du, gradient, value, derivative)
    derivative = gradient === nothing ? derivative : real(dot(gradient, du))
    if derivative === nothing
        computed_value, derivative = merit_ϕdϕ(ev, u, du, zero(eltype(u)))
        value = value === nothing ? computed_value : value
    end
    return (value === nothing ? merit_value!(ev.merit, ev, u) : value), derivative
end

function validate_armijo(c_1, contraction, initial_alpha, maxiters)
    0 < c_1 < 1 || throw(ArgumentError("c_1 must lie strictly between zero and one."))
    0 < contraction < 1 || throw(ArgumentError("contraction must lie strictly between zero and one."))
    isfinite(initial_alpha) && initial_alpha > 0 || throw(ArgumentError("initial_alpha must be positive and finite."))
    maxiters > 0 || throw(ArgumentError("maxiters must be positive."))
    return nothing
end
