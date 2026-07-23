module LineSearch

using ADTypes: ADTypes
using CommonSolve: CommonSolve
using ConcreteStructs: @concrete
using FastClosures: @closure
using LinearAlgebra: norm, dot
using MaybeInplace: @bb
using SciMLBase: SciMLBase, AbstractNonlinearProblem, OptimizationProblem,
                 ReturnCode, NonlinearFunction
using SciMLJacobianOperators: VecJacOperator, JacVecOperator
using StaticArraysCore: SArray

abstract type AbstractLineSearchAlgorithm end
abstract type AbstractLineSearchCache end

# Needed for certain algorithms like RobustNonMonotoneLineSearch
function callback_into_cache!(::AbstractLineSearchCache, _) end

# By default, reinit! does nothing
function SciMLBase.reinit!(::AbstractLineSearchCache; kwargs...) end

include("utils.jl")

include("backtracking.jl")
include("golden_section.jl")
include("li_fukushima.jl")
include("no_search.jl")
include("robust_non_monotone.jl")
include("strong_wolfe.jl")

include("line_searches_ext.jl")

"""
    LineSearchSolution(step_size, retcode)

The result returned by a line-search solve.

# Fields

- `step_size`: accepted step length for the current search direction.
- `retcode`: a `SciMLBase.ReturnCode` describing whether the line search found
  an acceptable step.

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
end

export LineSearchSolution

export BackTracking
export GoldenSection
export NoLineSearch, LiFukushimaLineSearch, RobustNonMonotoneLineSearch, StrongWolfeLineSearch
export LineSearchesJL

include("precompilation.jl")

end
