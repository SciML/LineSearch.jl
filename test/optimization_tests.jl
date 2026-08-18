using LineSearch, Test
using CommonSolve: init, solve!
using SciMLBase: OptimizationFunction, OptimizationProblem, ReturnCode

@testset "Strong Wolfe OptimizationProblem path" begin
    # Scalar Number u0 is supported on the static path.
    @testset "scalar Number u0" begin
        # φ(u) = ½(u-1)², minimum at u = 1; gradient via OptimizationFunction.grad
        f(u, p) = 0.5 * (u - 1)^2
        grad(u, p) = u - 1

        u0 = 0.0
        @test u0 isa Number

        optf = OptimizationFunction(f; grad)
        optprob = OptimizationProblem(optf, u0)
        cache = init(
            optprob, StrongWolfeLineSearch(; c2 = 0.1, α_init = 0.1, α_max = 4.0),
            f(u0, nothing), u0
        )

        @test cache isa LineSearch.StaticStrongWolfeLineSearchCache
        @test cache.mode isa LineSearch._ScalarObjective
        @test cache.grad_f === optf.grad

        du = -grad(u0, nothing)
        sol = solve!(cache, u0, du)
        @test sol.retcode == ReturnCode.Success
        @test sol.step_size ≈ 1.0

        sol_capped = solve!(cache, u0, du; α_max = 0.25)
        @test sol_capped.step_size <= 0.25 + 1.0e-12
    end

    @testset "missing OptimizationFunction.grad is rejected" begin
        optprob = OptimizationProblem((u, p) -> 0.5 * (u - 1)^2, 0.0)
        @test_throws ArgumentError init(
            optprob, StrongWolfeLineSearch(), 0.5, 0.0
        )
    end

    @testset "Vector u0" begin
        f(u, p) = sum(abs2, u)
        grad(u, p) = 2 .* u

        u0 = [1.0, 1.0]
        optf = OptimizationFunction(f; grad)
        optprob = OptimizationProblem(optf, u0)
        cache = init(
            optprob, StrongWolfeLineSearch(; c2 = 0.1, α_init = 1.0, α_max = 4.0),
            f(u0, nothing), u0
        )

        @test cache isa LineSearch.StaticStrongWolfeLineSearchCache
        @test cache.mode isa LineSearch._ScalarObjective
        @test cache.grad_f === optf.grad

        du = -grad(u0, nothing)
        sol = solve!(cache, u0, du)
        @test sol.retcode == ReturnCode.Success
        @test sol.step_size ≈ 0.5
    end
end
