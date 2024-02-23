using LineSearch
using Test

@testset "LineSearch.jl" begin
    # Write your tests here.

    # Test1: Quadratic function
    f(u) = u*u
    x = 1
    g = 2
    p = -g 
    α = 1

    # Create a Backtracking object
    my_b = Backtracking(x, g, p, f, α)
    α = Backtracking(my_b)

    print(α)

    # Test alpha
    @test α == 0.5
    @test 0 == 0
end
