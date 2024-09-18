module LineSearch

using ADTypes: ADTypes
using CommonSolve: CommonSolve
using ConcreteStructs: @concrete
using DifferentiationInterface: DifferentiationInterface
using FastClosures: @closure
using SciMLBase: AbstractNonlinearProblem, ReturnCode

const DI = DifferentiationInterface

abstract type AbstractLineSearchAlgorithm end
abstract type AbstractLineSearchCache end

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
