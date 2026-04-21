# Here we write out Newton Raphson and test integration with LineSearch.jl. Main tests are
# over at NonlinearSolve.jl and SimpleNonlinearSolve.jl
# Note: Enzyme tests are in a separate test group (test/enzyme/)
@testsetup module RootFinding

using SciMLBase, DifferentiationInterface, ForwardDiff
using SciMLBase: AbstractNonlinearProblem
const DI = DifferentiationInterface

function newton_raphson(prob::AbstractNonlinearProblem, ls)
    if SciMLBase.isinplace(prob)
        return newton_raphson_iip(prob, ls)
    else
        return newton_raphson_oop(prob, ls)
    end
end

function newton_raphson_oop(prob::AbstractNonlinearProblem, ls)
    u = copy(prob.u0)
    fu = prob.f(u, prob.p)

    ls_cache = init(prob, ls, fu, u)

    alphas = Float64[]
    iter = 0
    for _ in 1:100
        iter += 1

        maximum(abs, fu) < 1.0e-8 && return true, fu, u, iter, alphas

        J = DI.jacobian(prob.f, AutoForwardDiff(), u, Constant(prob.p))
        δu = -J \ fu

        ls_sol = solve!(ls_cache, u, δu)

        push!(alphas, ls_sol.step_size)
        @. u = u + ls_sol.step_size * δu

        fu = prob.f(u, prob.p)
    end

    return false, fu, u, iter, alphas
end

function newton_raphson_iip(prob::AbstractNonlinearProblem, ls)
    u = copy(prob.u0)
    fu = similar(u)
    fu2 = similar(u)
    prob.f(fu, u, prob.p)

    ls_cache = init(prob, ls, fu, u)

    alphas = Float64[]
    iter = 0
    for _ in 1:100
        iter += 1

        maximum(abs, fu) < 1.0e-8 && return true, fu, u, iter, alphas

        J = DI.jacobian(prob.f, fu2, AutoForwardDiff(), u, Constant(prob.p))
        δu = -J \ fu

        ls_sol = solve!(ls_cache, u, δu)

        push!(alphas, ls_sol.step_size)
        @. u = u + ls_sol.step_size * δu

        prob.f(fu, u, prob.p)
    end

    return false, fu, u, iter, alphas
end

export newton_raphson

end

@testitem "LineSearches.jl: Newton Raphson" tags = [:linesearchesjl] setup = [RootFinding] begin
    using LineSearches, SciMLBase
    using ADTypes, Tracker, ForwardDiff, Zygote, ReverseDiff, FiniteDiff

    @testset "OOP Problem" begin
        nlf(x, p) = x .^ 2 .- p
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

        @testset for autodiff in (
                AutoTracker(), AutoForwardDiff(), AutoZygote(),
                AutoReverseDiff(), AutoFiniteDiff(),
            )
            @testset "method: $(nameof(typeof(method)))" for method in (
                    LineSearches.BackTracking(; order = 3),
                    StrongWolfe(),
                    HagerZhang(),
                    MoreThuente(),
                    Static(),
                )
                linesearch = LineSearchesJL(; method, autodiff)
                converged, fu, u, iter, alphas = newton_raphson(nlp, linesearch)

                @test fu ≈ [0.0, 0.0] atol = 1.0e-3
                @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-3
            end
        end
    end

    @testset "In-Place Problem" begin
        nlf(dx, x, p) = (dx .= x .^ 2 .- p)
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

        @testset for autodiff in (
                AutoForwardDiff(), AutoReverseDiff(), AutoFiniteDiff(),
            )
            @testset "method: $(nameof(typeof(method)))" for method in (
                    LineSearches.BackTracking(; order = 3),
                    StrongWolfe(),
                    HagerZhang(),
                    MoreThuente(),
                    Static(),
                )
                linesearch = LineSearchesJL(; method, autodiff)
                converged, fu, u, iter, alphas = newton_raphson(nlp, linesearch)

                @test fu ≈ [0.0, 0.0] atol = 1.0e-3
                @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-3
            end
        end
    end
end

@testitem "Native Line Search: Newton Raphson" tags = [:core] setup = [RootFinding] begin
    using SciMLBase
    using ADTypes, Tracker, ForwardDiff, Zygote, ReverseDiff, FiniteDiff

    @testset "OOP Problem" begin
        nlf(x, p) = x .^ 2 .- p
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

        @testset "method: $(nameof(typeof(method)))" for method in (
                LiFukushimaLineSearch(),
                NoLineSearch(0.5),
                GoldenSection(; tol = 1.0e-4),
                RobustNonMonotoneLineSearch(),
                RobustNonMonotoneLineSearch(; M = 1), #strictly monotonous case
                RobustNonMonotoneLineSearch(; M = 15),
            )
            converged, fu, u, iter, alphas = newton_raphson(nlp, method)

            @test fu ≈ [0.0, 0.0] atol = 1.0e-1
            @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-1
        end

        @testset for autodiff in (
                AutoTracker(), AutoForwardDiff(), AutoZygote(),
                AutoReverseDiff(), AutoFiniteDiff(),
            )
            @testset "method: $(nameof(typeof(method)))" for method in (
                    BackTracking(; order = Val(3), autodiff),
                    BackTracking(; order = Val(2), autodiff),
                    StrongWolfeLineSearch(; autodiff),
                )
                converged, fu, u, iter, alphas = newton_raphson(nlp, method)

                @test fu ≈ [0.0, 0.0] atol = 1.0e-3
                @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-3
            end
        end
    end

    @testset "In-Place Problem" begin
        nlf(dx, x, p) = (dx .= x .^ 2 .- p)
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

        @testset "method: $(nameof(typeof(method)))" for method in (
                LiFukushimaLineSearch(),
                NoLineSearch(0.5),
                GoldenSection(; tol = 1.0e-4),
                RobustNonMonotoneLineSearch(),
                RobustNonMonotoneLineSearch(; M = 1), #strictly monotonous case
                RobustNonMonotoneLineSearch(; M = 15),
            )
            converged, fu, u, iter, alphas = newton_raphson(nlp, method)

            @test fu ≈ [0.0, 0.0] atol = 1.0e-1
            @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-1
        end

        @testset for autodiff in (
                AutoForwardDiff(), AutoReverseDiff(), AutoFiniteDiff(),
            )
            @testset "method: $(nameof(typeof(method)))" for method in (
                    BackTracking(; order = Val(3), autodiff),
                    BackTracking(; order = Val(2), autodiff),
                    StrongWolfeLineSearch(; autodiff),
                )
                converged, fu, u, iter, alphas = newton_raphson(nlp, method)

                @test fu ≈ [0.0, 0.0] atol = 1.0e-3
                @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-3
            end
        end
    end
end

@testitem "Native Strong Wolfe edge cases" tags = [:core] begin
    quadratic_eval(α) = (0.5 * (α - 1.0)^2, α - 1.0)

    @testset "initial convergence" begin
        ϕ_0, dϕ_0 = quadratic_eval(0.0)
        α, ok = LineSearch._sw_search(quadratic_eval, ϕ_0, dϕ_0, 1.0e-4, 0.9, 1.0, 4.0, 10, 10)

        @test ok
        @test α ≈ 1.0
    end

    @testset "initial need for bracketing" begin
        ϕ_0, dϕ_0 = quadratic_eval(0.0)
        α, ok = LineSearch._sw_search(
            quadratic_eval, ϕ_0, dϕ_0, 1.0e-4, 0.1, 0.25, 4.0, 10, 10
        )

        @test ok
        @test α ≈ 1.0
        @test α > 0.25
    end

    @testset "initial point is on the upward slope" begin
        uphill_eval(α) = (0.5 * (α + 1.0)^2, α + 1.0)
        ϕ_0, dϕ_0 = uphill_eval(0.0)
        α, ok = LineSearch._sw_search(uphill_eval, ϕ_0, dϕ_0, 1.0e-4, 0.9, 1.0, 4.0, 10, 10)

        @test !ok
        @test α == 0.0
    end

    @testset "initial trial has already passed the minimum" begin
        ϕ_0, dϕ_0 = quadratic_eval(0.0)
        α, ok = LineSearch._sw_search(
            quadratic_eval, ϕ_0, dϕ_0, 1.0e-4, 0.1, 3.0, 4.0, 10, 10
        )

        @test ok
        @test α ≈ 1.0
        @test α < 3.0
    end

    @testset "nonfinite trial values" begin
        nonfinite_eval(α) = α > 2.0 ? (Inf, Inf) : quadratic_eval(α)
        ϕ_0, dϕ_0 = nonfinite_eval(0.0)
        α, ok = LineSearch._sw_search(
            nonfinite_eval, ϕ_0, dϕ_0, 1.0e-4, 0.1, 3.0, 4.0, 10, 10
        )

        @test ok
        @test isfinite(α)
        @test 0.0 < α <= 2.0
    end
end
