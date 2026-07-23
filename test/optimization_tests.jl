using LineSearch, Test
using CommonSolve: init, solve!
using SciMLBase: OptimizationProblem, ReturnCode

@testset "Static Strong Wolfe OptimizationProblem path" begin
    # Scalar Number u0 is supported on the static path.
    @testset "scalar Number u0" begin
        # φ(u) = ½(u-1)², minimum at u = 1; grad_f = ∇φ
        f(u, p) = 0.5 * (u - 1)^2
        grad_f(u, p) = u - 1

        u0 = 0.0
        @test u0 isa Number

        optprob = OptimizationProblem(f, u0)
        cache = init(
            optprob, StrongWolfeLineSearch(; c2 = 0.1, α_init = 0.1, α_max = 4.0),
            f(u0, nothing), u0; grad_f
        )

        @test cache isa LineSearch.StaticStrongWolfeLineSearchCache
        @test cache.mode isa LineSearch._ScalarObjective

        du = -grad_f(u0, nothing)
        sol = solve!(cache, u0, du)
        @test sol.retcode == ReturnCode.Success
        @test sol.step_size ≈ 1.0

        sol_capped = solve!(cache, u0, du; α_max = 0.25)
        @test sol_capped.step_size <= 0.25 + 1.0e-12
    end

    @testset "non-static OptimizationProblem is rejected" begin
        f(u, p) = sum(abs2, u)
        optprob = OptimizationProblem(f, [0.0, 0.0])
        @test_throws ArgumentError init(
            optprob, StrongWolfeLineSearch(), [0.0], [0.0, 0.0];
            grad_f = (u, p) -> 2 .* u
        )
    end
end
