# Line searches driven by the objective merit (`OptimizationProblem`) rather
# than the residual merit (`NonlinearProblem`).
using LineSearch, Test
using SciMLBase, CommonSolve, LinearAlgebra
using SciMLBase: ReturnCode, NonlinearProblem, OptimizationProblem, OptimizationFunction
using ADTypes: AutoForwardDiff
import ForwardDiff

@testset "Objective merit" begin

    # ------------------------------------------------------------- problems

    rosen(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    function rosen_grad!(G, x, p)
        G[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
        G[2] = 200 * (x[2] - x[1]^2)
        return G
    end

    # Same problem stated both ways: F(u) = 0 with merit ‖F‖²/2, and the
    # objective f(u) = ‖F(u)‖²/2 directly.
    Fres(u, p) = [u[1]^2 - 2, u[2] - u[1]]
    Fobj(u, p) = sum(abs2, Fres(u, p)) / 2
    Fobj_grad!(G, u, p) = (ForwardDiff.gradient!(G, x -> Fobj(x, p), u); G)

    ALGS = (
        "StrongWolfe" => StrongWolfeLineSearch(),
        "BackTracking" => BackTracking(),
    )

    # --------------------------------------------------- objective entry point

    @testset "descends from an OptimizationProblem: $name" for (name, alg) in ALGS
        optf = OptimizationFunction(rosen; grad = rosen_grad!)
        prob = OptimizationProblem(optf, [-1.2, 1.0])

        u = [-1.2, 1.0]
        g = zeros(2)
        rosen_grad!(g, u, nothing)
        du = -g ./ norm(g, 1)

        cache = CommonSolve.init(prob, alg, u)
        sol = CommonSolve.solve!(cache, u, du)

        @test sol.retcode == ReturnCode.Success
        @test sol.step_size > 0
        @test rosen(u .+ sol.step_size .* du, nothing) < rosen(u, nothing)
    end

    @testset "GoldenSection is derivative free from an OptimizationProblem" begin
        # No gradient supplied at all: a derivative-free search must not demand
        # a Jacobian/AD backend.
        prob = OptimizationProblem(OptimizationFunction(rosen), [-1.2, 1.0])
        u = [-1.2, 1.0]
        du = [1.0, 0.0]
        cache = CommonSolve.init(prob, GoldenSection(), u)
        sol = CommonSolve.solve!(cache, u, du)
        @test sol.retcode == ReturnCode.Success
        @test rosen(u .+ sol.step_size .* du, nothing) < rosen(u, nothing)
    end

    # ---------------------------------------- the two merits agree numerically

    @testset "residual and objective merits agree: $name" for (name, alg) in ALGS
        u = [1.5, 1.0]
        fu = Fres(u, nothing)
        g = zeros(2)
        Fobj_grad!(g, u, nothing)
        du = -g

        nlprob = NonlinearProblem(Fres, u)
        c_res = CommonSolve.init(nlprob, alg, fu, u; autodiff = AutoForwardDiff())
        s_res = CommonSolve.solve!(c_res, u, du)

        optprob = OptimizationProblem(
            OptimizationFunction(Fobj; grad = Fobj_grad!), u
        )
        c_obj = CommonSolve.init(optprob, alg, u)
        s_obj = CommonSolve.solve!(c_obj, u, du)

        @test s_res.retcode == s_obj.retcode
        # ϕ(α) = ‖F‖²/2 is literally the same scalar function in both cases, so
        # the algorithms must take the identical path.
        @test s_res.step_size ≈ s_obj.step_size rtol = 1.0e-10
    end

    # ------------------------------------------- solution carries ϕ and dϕ

    @testset "LineSearchSolution reports ϕ and dϕ" begin
        optf = OptimizationFunction(rosen; grad = rosen_grad!)
        prob = OptimizationProblem(optf, [-1.2, 1.0])
        u = [-1.2, 1.0]
        g = zeros(2)
        rosen_grad!(g, u, nothing)
        du = -g ./ norm(g, 1)

        cache = CommonSolve.init(prob, StrongWolfeLineSearch(), u)
        sol = CommonSolve.solve!(cache, u, du)
        un = u .+ sol.step_size .* du
        gn = zeros(2)
        rosen_grad!(gn, un, nothing)
        # The gradient at the accepted step is left in the cache, so the caller
        # need not re-evaluate it.
        @test cache.merit_eval.fu_cache ≈ gn
    end

    @testset "two-argument constructor stays available" begin
        sol = LineSearchSolution(0.5, ReturnCode.Success)
        @test sol.step_size == 0.5
        @test sol.retcode == ReturnCode.Success
        @test sol.ϕ === nothing
        @test sol.dϕ === nothing
    end

    # ------------------------------------------------ gradient arity handling

    @testset "accepts both 3-arg and 2-arg gradient closures" begin
        u = [-1.2, 1.0]
        g = zeros(2)
        rosen_grad!(g, u, nothing)
        du = -g ./ norm(g, 1)

        # Three-argument form, as written on a raw OptimizationFunction.
        p3 = OptimizationProblem(OptimizationFunction(rosen; grad = rosen_grad!), u)
        s3 = CommonSolve.solve!(
            CommonSolve.init(p3, StrongWolfeLineSearch(), u), u, du
        )

        # Two-argument form, as produced by `instantiate_function`.
        p2 = OptimizationProblem(
            OptimizationFunction(rosen; grad = (G, x) -> rosen_grad!(G, x, nothing)), u
        )
        s2 = CommonSolve.solve!(
            CommonSolve.init(p2, StrongWolfeLineSearch(), u), u, du
        )

        @test s3.retcode == ReturnCode.Success
        @test s2.step_size ≈ s3.step_size
    end

    @testset "missing gradient is reported clearly" begin
        prob = OptimizationProblem(OptimizationFunction(rosen), [-1.2, 1.0])
        @test_throws ArgumentError CommonSolve.init(
            prob, StrongWolfeLineSearch(), [-1.2, 1.0]
        )
    end

    # ---------------------------------------------------------- allocations

    @testset "solve! does not allocate in steady state" begin
        optf = OptimizationFunction(rosen; grad = rosen_grad!)
        prob = OptimizationProblem(optf, [-1.2, 1.0])
        u = [-1.2, 1.0]
        g = zeros(2)
        rosen_grad!(g, u, nothing)
        du = -g ./ norm(g, 1)

        for alg in (StrongWolfeLineSearch(), BackTracking())
            cache = CommonSolve.init(prob, alg, u)
            CommonSolve.solve!(cache, u, du)          # warm up
            a1 = @allocated CommonSolve.solve!(cache, u, du)
            a2 = @allocated CommonSolve.solve!(cache, u, du)
            @test a1 == a2
            @test a2 == 0
        end
    end

    # -------------------------------------------------------------- numerics

    @testset "Float32 and BigFloat" begin
        for T in (Float32, BigFloat)
            u = T[-1.2, 1.0]
            g = zeros(T, 2)
            rosen_grad!(g, u, nothing)
            du = -g ./ norm(g, 1)
            prob = OptimizationProblem(
                OptimizationFunction(rosen; grad = rosen_grad!), u
            )
            sol = CommonSolve.solve!(
                CommonSolve.init(prob, StrongWolfeLineSearch(), u), u, du
            )
            @test sol.retcode == ReturnCode.Success
            @test sol.step_size isa T
            @test rosen(u .+ sol.step_size .* du, nothing) < rosen(u, nothing)
        end
    end
end
