using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    f_oop = (u, p) -> u .^ 2 .- p
    f_iip = (fu, u, p) -> (fu .= u .^ 2 .- p)
    u0 = [1.0, 1.0]
    p = [2.0]
    fu = f_oop(u0, p)

    prob_oop = SciMLBase.NonlinearProblem(
        SciMLBase.NonlinearFunction{false}(f_oop), u0, p
    )
    prob_iip = SciMLBase.NonlinearProblem(
        SciMLBase.NonlinearFunction{true}(f_iip), u0, p
    )

    @compile_workload begin
        for prob in (prob_oop, prob_iip)
            u = copy(prob.u0)
            fu_work = copy(fu)
            du = -fu_work

            # NoLineSearch
            ls = NoLineSearch()
            cache = CommonSolve.init(prob, ls, fu_work, u)
            CommonSolve.solve!(cache, u, du)

            # LiFukushimaLineSearch
            ls = LiFukushimaLineSearch()
            cache = CommonSolve.init(prob, ls, fu_work, u)
            CommonSolve.solve!(cache, u, du)

            # RobustNonMonotoneLineSearch
            ls = RobustNonMonotoneLineSearch()
            cache = CommonSolve.init(prob, ls, fu_work, u)
            CommonSolve.solve!(cache, u, du)
            callback_into_cache!(cache, fu_work)
        end
    end
end
