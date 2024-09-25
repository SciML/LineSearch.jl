module LineSearch

using ADTypes: ADTypes
using CommonSolve: CommonSolve
using ConcreteStructs: @concrete
using DifferentiationInterface: DifferentiationInterface
using EnzymeCore: EnzymeCore
using FastClosures: @closure
using LinearAlgebra: norm, dot
using MaybeInplace: @bb
using SciMLBase: SciMLBase, AbstractSciMLProblem, AbstractNonlinearProblem,
                 NonlinearProblem, NonlinearLeastSquaresProblem, NonlinearFunction,
                 ReturnCode
using SciMLJacobianOperators: VecJacOperator, JacVecOperator

const DI = DifferentiationInterface

abstract type AbstractLineSearchAlgorithm end
abstract type AbstractLineSearchCache end

# Needed for certain algorithms like RobustNonMonotoneLineSearch
function callback_into_cache!(cache::AbstractLineSearchCache, fu) end

# TODO: define `reinit!` for LineSearch

include("utils.jl")

include("li_fukushima.jl")
include("no_search.jl")
include("robust_non_monotone.jl")

include("line_searches_ext.jl")

@concrete struct LineSearchSolution
    step_size <: Real
    retcode::ReturnCode.T
end

export LineSearchSolution

export NoLineSearch, LiFukushimaLineSearch, RobustNonMonotoneLineSearch
export LineSearchesJL

end
