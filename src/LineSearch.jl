module LineSearch

using ADTypes: ADTypes
using CommonSolve: CommonSolve
using ConcreteStructs: @concrete
using DifferentiationInterface: DifferentiationInterface
using FastClosures: @closure

const DI = DifferentiationInterface

abstract type AbstractLineSearchAlgorithm end

include("li_fukushima.jl")
include("no_search.jl")
include("robust_non_monotone.jl")

include("line_searches_ext.jl")

@concrete struct LineSearchSolution
    alpha <: Real
    success::Bool
end

export LineSearchSolution

end
