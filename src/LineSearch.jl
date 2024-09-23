module LineSearch

using ADTypes: ADTypes
using CommonSolve: CommonSolve
using ConcreteStructs: @concrete
using DifferentiationInterface: DifferentiationInterface
using EnzymeCore: EnzymeCore
using FastClosures: @closure
using SciMLBase: AbstractSciMLProblem, AbstractNonlinearProblem, NonlinearProblem,
                 NonlinearLeastSquaresProblem, ReturnCode

const DI = DifferentiationInterface

abstract type AbstractLineSearchAlgorithm end
abstract type AbstractLineSearchCache end

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

export NoLineSearch, NoLineSearchCache

end
