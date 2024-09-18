@concrete struct LineSearchesJL <: AbstractLineSearchAlgorithm
    method
    initial_alpha
    autodiff <: Union{Nothing, ADTypes.AbstractADType}
end
