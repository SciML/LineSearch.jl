using LineSearch
using Test

@testset "LineSearch.jl" begin
    # Write your tests here.
    # Test1: Quadratic function
    f(u) = u'u
    x = [1.0]
    g = [2.0]
    p = -g 
    α = 1.0

    # Backtracking line search
    α = Backtracking(x, g, p, f, α)

    # print(α) # α=0.5

    # Test alpha
    @test α == 0.5
end
