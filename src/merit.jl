"""
    AbstractMerit

Supertype for the scalar function a line search reduces along the ray
`u + α ⋅ du`.

A line search algorithm is merit-agnostic: it consumes `ϕ(α)` and `ϕ'(α)` and
knows nothing about where they came from. The merit is the only thing that
differs between the two callers:

| merit | `ϕ(α)` | `ϕ'(α)` |
|:--|:--|:--|
| [`ResidualMerit`](@ref) (root finding) | `‖F(u + α du)‖² / 2` | `⟨F, J ⋅ du⟩` |
| [`ObjectiveMerit`](@ref) (optimization) | `f(u + α du)` | `⟨∇f, du⟩` |

Keeping this distinction out of the algorithms is what lets one implementation
of Hager–Zhang or Moré–Thuente serve both `NonlinearProblem`s and
`OptimizationProblem`s.
"""
abstract type AbstractMerit end

"""
    ResidualMerit()

Merit `ϕ(α) = ‖F(u + α du)‖² / 2` for root finding. The directional derivative
`⟨F, J ⋅ du⟩` requires a Jacobian-vector product.
"""
struct ResidualMerit <: AbstractMerit end

"""
    ObjectiveMerit()

Merit `ϕ(α) = f(u + α du)` for optimization, where `f` is the objective itself.

The directional derivative is `⟨∇f, du⟩` — a dot product against a gradient the
optimizer has typically already computed, rather than the Hessian-vector product
that [`ResidualMerit`](@ref) would require if pointed at `∇f = 0`.
"""
struct ObjectiveMerit <: AbstractMerit end

"""
    MeritEvaluator

Evaluates `ϕ(α)` and `(ϕ(α), ϕ'(α))` along the ray `u + α ⋅ du`, reusing
preallocated buffers. Built once by [`init_merit`](@ref) and reused across
`solve!` calls, so the steady-state line search allocates nothing.

`fu_cache` holds the residual under [`ResidualMerit`](@ref) and the gradient
under [`ObjectiveMerit`](@ref); in both cases it is left holding the quantity at
the most recently evaluated `α`, which lets the caller reuse it.
"""
@concrete mutable struct MeritEvaluator
    merit <: AbstractMerit
    f
    p
    deriv_op
    u_cache
    fu_cache
    stats <: Union{SciMLBase.NLStats, Nothing}
    last_α
    last_ϕ
    last_dϕ
    last_has_deriv::Bool
end

"""
    init_merit(prob, fu_or_u, u; autodiff = nothing, stats = nothing)

Build a [`MeritEvaluator`](@ref) for `prob`.

For an `AbstractNonlinearProblem` this yields [`ResidualMerit`](@ref) and needs
`fu` to size the residual cache. For an `OptimizationProblem` it yields
[`ObjectiveMerit`](@ref) and needs only `u`.
"""
function init_merit(
        prob::AbstractNonlinearProblem, fu, u;
        autodiff = nothing, stats::Union{SciMLBase.NLStats, Nothing} = nothing,
        need_deriv::Bool = true
    )
    # Derivative-free searches must not be forced to build a Jacobian operator,
    # which would demand an AD backend they never use.
    deriv_op = if need_deriv
        last(construct_jvp_or_vjp_operator(prob, fu, u; autodiff))
    else
        nothing
    end
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    nan = convert(promote_type(eltype(fu), eltype(u)), NaN)
    return MeritEvaluator(
        ResidualMerit(), prob.f, prob.p, deriv_op, u_cache, fu_cache, stats,
        nan, nan, nan, false
    )
end

function init_merit(
        prob::OptimizationProblem, u;
        autodiff = nothing, stats::Union{SciMLBase.NLStats, Nothing} = nothing,
        need_deriv::Bool = true
    )
    value, fg = if need_deriv
        objective_and_fused_gradient(prob.f, prob.p, u)
    else
        (@closure((x, p) -> prob.f.f(x, p)), nothing)
    end
    @bb u_cache = similar(u)
    @bb fu_cache = similar(u)
    nan = convert(real(eltype(u)), NaN)
    return MeritEvaluator(
        ObjectiveMerit(), value, prob.p, fg, u_cache, fu_cache, stats,
        nan, nan, nan, false
    )
end

# Symmetry with the nonlinear signature, where the second argument sizes the
# residual cache; for an objective it carries no information.
function init_merit(prob::OptimizationProblem, ::Any, u; kwargs...)
    return init_merit(prob, u; kwargs...)
end

function objective_and_fused_gradient(f::SciMLBase.AbstractOptimizationFunction, p, u)
    value = @closure (u, p) -> f.f(u, p)

    if f.fg !== nothing
        fg = f.fg
        applicable(fg, u, u, p) && return value, @closure((G, u, p) -> fg(G, u, p))
        return value, @closure(
                (G, u, p) -> begin
                    ϕ, grad = fg(u, p)
                    copyto!(G, grad)
                    return ϕ
                end
            )
    end

    if f.grad !== nothing
        g = f.grad
        if applicable(g, u, u, p)
            return value, @closure((G, u, p) -> (g(G, u, p); f.f(u, p)))
        end
        return value, @closure(
                (G, u, p) -> begin
                    copyto!(G, g(u, p))
                    return f.f(u, p)
                end
            )
    end

    throw(
        ArgumentError(
            "`ObjectiveMerit` needs a gradient. Supply an `OptimizationFunction` with an \
             analytic `grad`/`fg`, or one instantiated against an AD backend by \
             `OptimizationBase.instantiate_function`."
        )
    )
end

"""
    invalidate!(ev)

Forget which `α` was last evaluated. Line searches must call this on entry:
`u` and `du` change between calls, so a cached `α` from the previous search
refers to a different point entirely.
"""
function invalidate!(ev::MeritEvaluator)
    ev.last_α = oftype(ev.last_α, NaN)
    ev.last_has_deriv = false
    return ev
end

"""
    merit_ϕdϕ_at_zero(ev, u, du, ϕ0, dϕ0)

`ϕ(0)` and `ϕ'(0)`, taken from the caller when supplied.

An optimizer already holds the objective and gradient at the current iterate, so
making it pay for another evaluation costs a full derivative pass per outer
iteration. Passing them in through `solve!` avoids that.
"""
@inline function merit_ϕdϕ_at_zero(ev::MeritEvaluator, u, du, ϕ0, dϕ0)
    (ϕ0 === nothing || dϕ0 === nothing) && return merit_ϕdϕ(ev, u, du, zero(eltype(u)))
    return (ϕ0, dϕ0)
end

"""
    merit_ϕ(ev, u, du, α)

Evaluate `ϕ(α)` only. Cheaper than [`merit_ϕdϕ`](@ref) under
[`ObjectiveMerit`](@ref), where it skips the gradient entirely.
"""
merit_ϕ(ev::MeritEvaluator, u, du, α) = _merit_ϕ(ev.merit, ev, u, du, α)

"""
    merit_ϕdϕ(ev, u, du, α)

Evaluate `(ϕ(α), ϕ'(α))` in one pass.
"""
merit_ϕdϕ(ev::MeritEvaluator, u, du, α) = _merit_ϕdϕ(ev.merit, ev, u, du, α)

@inline function ray_point!(ev::MeritEvaluator, u, du, α, has_deriv::Bool)
    u_cache = ev.u_cache
    @bb @. u_cache = u + α * du
    ev.last_α = α
    ev.last_has_deriv = has_deriv
    return u_cache
end

"""
    ensure_evaluated_at!(ev, u, du, α)

Guarantee that `ev.u_cache` and `ev.fu_cache` hold the iterate and the
residual/gradient at `α`, re-evaluating only when the last probe was somewhere
else. Line searches call this before returning, so a caller can always reuse the
accepted point instead of recomputing it.

Returns the cached `(ϕ, ϕ')` values.
"""
function ensure_evaluated_at!(ev::MeritEvaluator, u, du, α)
    (ev.last_has_deriv && ev.last_α == α) || merit_ϕdϕ(ev, u, du, α)
    return (ev.last_ϕ, ev.last_dϕ)
end

function ensure_value_at!(ev::MeritEvaluator, u, du, α)
    ev.last_α == α || merit_ϕ(ev, u, du, α)
    return ev.last_ϕ
end

function solution_at!(ev::MeritEvaluator, u, du, α, retcode)
    ϕ, dϕ = ensure_evaluated_at!(ev, u, du, α)
    return LineSearchSolution(α, retcode, ϕ, dϕ)
end

function _merit_ϕ(::ResidualMerit, ev::MeritEvaluator, u, du, α)
    u_cache = ray_point!(ev, u, du, α, false)
    ev.fu_cache = evaluate_f!!(ev.f, ev.fu_cache, u_cache, ev.p)
    add_nf!(ev.stats)
    ϕ = @fastmath norm(ev.fu_cache)^2 / 2
    ev.last_ϕ = ϕ
    return ϕ
end

function _merit_ϕdϕ(::ResidualMerit, ev::MeritEvaluator, u, du, α)
    ev.deriv_op === nothing && throw(
        ArgumentError("this merit was built without a derivative operator")
    )
    u_cache = ray_point!(ev, u, du, α, true)
    ev.fu_cache = evaluate_f!!(ev.f, ev.fu_cache, u_cache, ev.p)
    add_nf!(ev.stats)
    dϕ = ev.deriv_op(du, u_cache, ev.fu_cache, ev.p)
    ϕ = @fastmath norm(ev.fu_cache)^2 / 2
    ev.last_ϕ = ϕ
    ev.last_dϕ = dϕ
    return (ϕ, dϕ)
end

function _merit_ϕ(::ObjectiveMerit, ev::MeritEvaluator, u, du, α)
    u_cache = ray_point!(ev, u, du, α, false)
    add_nf!(ev.stats)
    ϕ = ev.f(u_cache, ev.p)
    ev.last_ϕ = ϕ
    return ϕ
end

function _merit_ϕdϕ(::ObjectiveMerit, ev::MeritEvaluator, u, du, α)
    u_cache = ray_point!(ev, u, du, α, true)
    add_nf!(ev.stats)
    ϕ = ev.deriv_op(ev.fu_cache, u_cache, ev.p)
    dϕ = dot(ev.fu_cache, du)
    ev.last_ϕ = ϕ
    ev.last_dϕ = dϕ
    return (ϕ, dϕ)
end

function SciMLBase.reinit!(ev::MeritEvaluator; p = missing, stats = missing, kwargs...)
    p !== missing && (ev.p = p)
    stats !== missing && (ev.stats = stats)
    return ev
end
