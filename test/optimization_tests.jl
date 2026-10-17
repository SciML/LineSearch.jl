using LineSearch, Test
using StaticArrays: SVector
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
        @test sol_capped.retcode == ReturnCode.Success
        @test sol_capped.step_size == 0.25
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

    # Minimizer outside the box [0, 1]²; the ray hits the boundary at α = 0.2.
    @testset "minimizer outside box: accept α_max" begin
        c = SVector(3.0, 3.0)
        f(u, p) = sum(abs2, u .- c) / 2
        grad(u, p) = u .- c

        u0 = SVector(0.5, 0.2)
        du = -grad(u0, nothing)
        α_box = minimum((1 .- u0) ./ du)
        @test α_box ≈ 0.2

        optprob = OptimizationProblem(OptimizationFunction(f; grad), u0)
        @testset "α_init = $α_init" for α_init in (1.0, 0.2, 0.03)
            cache = init(
                optprob, StrongWolfeLineSearch(; c2 = 0.1, α_init),
                f(u0, nothing), u0
            )
            sol = solve!(cache, u0, du; α_max = α_box)
            @test sol.retcode == ReturnCode.Success
            @test sol.step_size == α_box
            @test f(u0 + sol.step_size * du, nothing) < f(u0, nothing)
        end

        g!(G, u, p) = (G .= u .- c; G)
        uv, duv = Vector(u0), Vector(du)
        vprob = OptimizationProblem(OptimizationFunction(f; grad = g!), uv)
        vcache = init(
            vprob, StrongWolfeLineSearch(; c2 = 0.1, α_init = 0.03, α_max = α_box), uv
        )
        vsol = solve!(vcache, uv, duv)
        @test vsol.retcode == ReturnCode.Success
        @test vsol.step_size == α_box
    end

    @testset "Armijo failure at α_max still zooms" begin
        f(u, p) = 0.5 * (u - 1)^2
        grad(u, p) = u - 1
        optprob = OptimizationProblem(OptimizationFunction(f; grad), 0.0)
        cache = init(
            optprob, StrongWolfeLineSearch(; c2 = 0.1, α_init = 3.0), 0.5, 0.0
        )
        sol = solve!(cache, 0.0, 1.0; α_max = 3.0)
        @test sol.retcode == ReturnCode.Success
        @test sol.step_size < 3.0
        @test abs(grad(sol.step_size, nothing)) <= 0.1
    end
end
