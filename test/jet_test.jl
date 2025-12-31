@testitem "JET static analysis" begin
    using JET, LineSearch, SciMLBase, CommonSolve, ADTypes

    f_oop(u, p) = u .^ 2 .- p
    u0 = [1.0, 2.0]
    p = [0.5, 0.5]
    prob = NonlinearProblem(f_oop, u0, p)
    fu = f_oop(u0, p)

    @testset "Constructor type stability" begin
        rep = JET.report_call(BackTracking, ())
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(LiFukushimaLineSearch, ())
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(NoLineSearch, ())
        @test length(JET.get_reports(rep)) == 0

        rep = JET.report_call(RobustNonMonotoneLineSearch, ())
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "init type stability" begin
        autodiff = AutoForwardDiff()
        alg_bt = BackTracking(; autodiff)
        rep = JET.@report_opt target_modules = (LineSearch,) CommonSolve.init(
            prob, alg_bt, fu, u0)
        @test length(JET.get_reports(rep)) == 0

        alg_lf = LiFukushimaLineSearch()
        rep = JET.@report_opt target_modules = (LineSearch,) CommonSolve.init(
            prob, alg_lf, fu, u0)
        @test length(JET.get_reports(rep)) == 0

        alg_no = NoLineSearch()
        rep = JET.@report_opt target_modules = (LineSearch,) CommonSolve.init(
            prob, alg_no, fu, u0)
        @test length(JET.get_reports(rep)) == 0

        alg_robust = RobustNonMonotoneLineSearch()
        rep = JET.@report_opt target_modules = (LineSearch,) CommonSolve.init(
            prob, alg_robust, fu, u0)
        @test length(JET.get_reports(rep)) == 0
    end
end
