using LineSearch
using Test

@testset "LineSearch.jl" begin
    # Write your tests here.

    # Quadratic function
    f(u) = u*u
    u0 = 1
    g = 2
    p = -g
    α = 1

    # Create a Backtracking object
    my_b = Backtracking(x, g, p, f, α)
    α = Backtracking(my_b)

    # Test alpha
    @test α == 0
end

# t = Template(;
#             user = "Xiaoyi-Qu",
#             license = "MIT",
#             authors = ["To be added"],
#             plugins = [
#                 TravisCI(),
#                 Codecov(),
#                 Coveralls(),
#                 AppVeyor()
#             ],
#     )

# generate("LineSearch", t)