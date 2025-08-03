module LineSearch

using ADTypes: ADTypes
using CommonSolve: CommonSolve
using ConcreteStructs: @concrete
using FastClosures: @closure
using LinearAlgebra: norm, dot
using MaybeInplace: @bb
using SciMLBase: SciMLBase, AbstractSciMLProblem, AbstractNonlinearProblem, ReturnCode,
                 NonlinearProblem, NonlinearLeastSquaresProblem, NonlinearFunction
using SciMLJacobianOperators: VecJacOperator, JacVecOperator
using StaticArraysCore: SArray

# The 2 Core abstractions of this module
# Base type for all line search algorithm
abstract type AbstractLineSearchAlgorithm end 
# Base type for all algorithm-specific caching
abstract type AbstractLineSearchCache end

# Needed for certain algorithms like RobustNonMonotoneLineSearch
function callback_into_cache!(::AbstractLineSearchCache, _) end

# By default, reinit! does nothing
function SciMLBase.reinit!(::AbstractLineSearchCache; kwargs...) end

include("utils.jl")

include("backtracking.jl")
include("li_fukushima.jl")
include("no_search.jl")
include("robust_non_monotone.jl")

include("line_searches_ext.jl")
include("hager_zhang.jl")

@concrete struct LineSearchSolution
    step_size
    retcode::ReturnCode.T
end

export LineSearchSolution

export BackTracking
export HagerZhang
export NoLineSearch, LiFukushimaLineSearch, RobustNonMonotoneLineSearch
export LineSearchesJL

end
