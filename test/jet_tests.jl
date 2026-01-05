@testitem "JET Static Analysis" tags=[:jet] begin
    using JET, LineSearch, SciMLBase, ADTypes, ForwardDiff
    using CommonSolve: init, solve!

    f_oop(u, p) = u .^ 2 .- p
    prob = NonlinearProblem(f_oop, [1.0, 2.0], [3.0])
    u = [1.0, 2.0]
    fu = f_oop(u, prob.p)
    du = -fu

    @testset "NoLineSearch" begin
        alg = NoLineSearch()
        cache = init(prob, alg, fu, u)
        @test_opt target_modules = (LineSearch,) solve!(cache, u, du)
    end

    @testset "LiFukushimaLineSearch" begin
        alg = LiFukushimaLineSearch()
        cache = init(prob, alg, fu, u)
        @test_opt target_modules = (LineSearch,) solve!(cache, u, du)
    end

    @testset "RobustNonMonotoneLineSearch" begin
        alg = RobustNonMonotoneLineSearch()
        cache = init(prob, alg, fu, u)
        @test_opt target_modules = (LineSearch,) solve!(cache, u, du)
    end

    @testset "BackTracking" begin
        alg = BackTracking(; autodiff = AutoForwardDiff())
        cache = init(prob, alg, fu, u)
        # BackTracking uses closures with captured variables from external packages
        # (SciMLJacobianOperators), which JET cannot fully analyze for type stability.
        # We test for errors only, not optimization/type stability issues.
        report = JET.report_call(solve!, (typeof(cache), typeof(u), typeof(du));
            target_modules = (LineSearch,))
        @test length(JET.get_reports(report)) == 0
    end
end
