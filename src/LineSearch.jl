module LineSearch

using ADTypes: ADTypes
using CommonSolve: CommonSolve
using ConcreteStructs: @concrete
using FastClosures: @closure
using LinearAlgebra: norm, dot
using MaybeInplace: @bb
using SciMLBase: SciMLBase, AbstractNonlinearProblem, ReturnCode, NonlinearFunction,
    OptimizationProblem
using SciMLJacobianOperators: VecJacOperator, JacVecOperator
using StaticArraysCore: SArray

abstract type AbstractLineSearchAlgorithm end
abstract type AbstractLineSearchCache end

# Needed for certain algorithms like RobustNonMonotoneLineSearch
function callback_into_cache!(::AbstractLineSearchCache, _) end

"""
    set_initial_step!(cache, α)

Set the first trial step length used by the next `solve!`.

Quasi-Newton methods need this per iteration: the unit step is the right first
guess once curvature information has accumulated, but not on the first
iteration, where the direction is plain steepest descent and has no natural
scale. Algorithms that ignore the initial step leave this a no-op.
"""
set_initial_step!(cache::AbstractLineSearchCache, α) = cache

# By default, reinit! does nothing
function SciMLBase.reinit!(::AbstractLineSearchCache; kwargs...) end

include("utils.jl")
include("merit.jl")

include("backtracking.jl")
include("hager_zhang.jl")
include("more_thuente.jl")
include("golden_section.jl")
include("li_fukushima.jl")
include("no_search.jl")
include("robust_non_monotone.jl")
include("strong_wolfe.jl")

include("line_searches_ext.jl")

"""
    LineSearchSolution(step_size, retcode)
    LineSearchSolution(step_size, retcode, ϕ, dϕ)

The result returned by a line-search solve.

# Fields

- `step_size`: accepted step length for the current search direction.
- `retcode`: a `SciMLBase.ReturnCode` describing whether the line search found
  an acceptable step.
- `ϕ`: merit value at `step_size`, or `nothing` if the algorithm did not report
  it. Returning it lets the caller reuse the accepted point instead of
  re-evaluating the objective, which for an AD-defined problem is a full
  derivative pass per outer iteration.
- `dϕ`: directional derivative at `step_size`, or `nothing`.

# Examples

```julia
using LineSearch
using SciMLBase

sol = LineSearchSolution(0.5, SciMLBase.ReturnCode.Success)
sol.step_size
sol.retcode
```
"""
@concrete struct LineSearchSolution
    step_size
    retcode::ReturnCode.T
    ϕ
    dϕ
end

function LineSearchSolution(step_size, retcode)
    return LineSearchSolution(step_size, retcode, nothing, nothing)
end

export LineSearchSolution
export set_initial_step!

export BackTracking
export GoldenSection
export NoLineSearch, LiFukushimaLineSearch, RobustNonMonotoneLineSearch, StrongWolfeLineSearch
export HagerZhangLineSearch, MoreThuenteLineSearch
export AbstractMerit, ResidualMerit, ObjectiveMerit
export LineSearchesJL

include("precompilation.jl")

end
