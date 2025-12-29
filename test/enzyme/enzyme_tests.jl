# Enzyme-specific tests for LineSearch.jl
# This file contains tests that use Enzyme for automatic differentiation
# Separated from main test suite due to Julia version compatibility issues

using Test, SafeTestsets
using LinearAlgebra

@safetestset "Enzyme: Root Finding Tests" begin
    using SciMLBase, ForwardDiff
    import DifferentiationInterface as DI
    using LineSearch, LineSearches, Test
    using ADTypes, Enzyme, ReverseDiff, FiniteDiff

    function newton_raphson(prob, ls)
        if SciMLBase.isinplace(prob)
            return newton_raphson_iip(prob, ls)
        else
            return newton_raphson_oop(prob, ls)
        end
    end

    function newton_raphson_oop(prob, ls)
        u = copy(prob.u0)
        fu = prob.f(u, prob.p)
        ls_cache = init(prob, ls, fu, u)
        alphas = Float64[]
        iter = 0
        for _ in 1:100
            iter += 1
            maximum(abs, fu) < 1e-8 && return true, fu, u, iter, alphas
            J = DI.jacobian(prob.f, AutoForwardDiff(), u, DI.Constant(prob.p))
            δu = -J \ fu
            ls_sol = solve!(ls_cache, u, δu)
            push!(alphas, ls_sol.step_size)
            @. u = u + ls_sol.step_size * δu
            fu = prob.f(u, prob.p)
        end
        return false, fu, u, iter, alphas
    end

    function newton_raphson_iip(prob, ls)
        u = copy(prob.u0)
        fu = similar(u)
        fu2 = similar(u)
        prob.f(fu, u, prob.p)
        ls_cache = init(prob, ls, fu, u)
        alphas = Float64[]
        iter = 0
        for _ in 1:100
            iter += 1
            maximum(abs, fu) < 1e-8 && return true, fu, u, iter, alphas
            J = DI.jacobian(prob.f, fu2, AutoForwardDiff(), u, DI.Constant(prob.p))
            δu = -J \ fu
            ls_sol = solve!(ls_cache, u, δu)
            push!(alphas, ls_sol.step_size)
            @. u = u + ls_sol.step_size * δu
            prob.f(fu, u, prob.p)
        end
        return false, fu, u, iter, alphas
    end

    @testset "LineSearches.jl with AutoEnzyme" begin
        @testset "OOP Problem" begin
            nlf(x, p) = x .^ 2 .- p
            nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

            @testset "method: $(nameof(typeof(method)))" for method in (
                LineSearches.BackTracking(; order = 3),
                StrongWolfe(),
                HagerZhang(),
                MoreThuente(),
                Static()
            )
                linesearch = LineSearchesJL(; method, autodiff = AutoEnzyme())
                converged, fu, u, iter, alphas = newton_raphson(nlp, linesearch)

                @test fu≈[0.0, 0.0] atol=1e-3
                @test abs.(u)≈sqrt.([3.0, 3.0]) atol=1e-3
            end
        end

        @testset "In-Place Problem" begin
            nlf(dx, x, p) = (dx .= x .^ 2 .- p)
            nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

            @testset "method: $(nameof(typeof(method)))" for method in (
                LineSearches.BackTracking(; order = 3),
                StrongWolfe(),
                HagerZhang(),
                MoreThuente(),
                Static()
            )
                linesearch = LineSearchesJL(; method, autodiff = AutoEnzyme())
                converged, fu, u, iter, alphas = newton_raphson(nlp, linesearch)

                @test fu≈[0.0, 0.0] atol=1e-3
                @test abs.(u)≈sqrt.([3.0, 3.0]) atol=1e-3
            end
        end
    end

    @testset "Native Line Search with AutoEnzyme" begin
        @testset "OOP Problem" begin
            nlf(x, p) = x .^ 2 .- p
            nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

            @testset "method: $(nameof(typeof(method)))" for method in (
                BackTracking(; order = Val(3), autodiff = AutoEnzyme()),
                BackTracking(; order = Val(2), autodiff = AutoEnzyme())
            )
                converged, fu, u, iter, alphas = newton_raphson(nlp, method)

                @test fu≈[0.0, 0.0] atol=1e-3
                @test abs.(u)≈sqrt.([3.0, 3.0]) atol=1e-3
            end
        end

        @testset "In-Place Problem" begin
            nlf(dx, x, p) = (dx .= x .^ 2 .- p)
            nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

            @testset "method: $(nameof(typeof(method)))" for method in (
                BackTracking(; order = Val(3), autodiff = AutoEnzyme()),
                BackTracking(; order = Val(2), autodiff = AutoEnzyme())
            )
                converged, fu, u, iter, alphas = newton_raphson(nlp, method)

                @test fu≈[0.0, 0.0] atol=1e-3
                @test abs.(u)≈sqrt.([3.0, 3.0]) atol=1e-3
            end
        end
    end
end

@safetestset "Enzyme: Custom Optimizer Tests" begin
    using LinearAlgebra, SciMLBase, LineSearch, SciMLJacobianOperators, Test
    using LineSearches
    using ADTypes, Enzyme, ReverseDiff, FiniteDiff, ForwardDiff

    function gradient_descent(
            prob, alg; g_atol::Real = 1e-5, maxiters::Int = 10000, autodiff = nothing)
        u = copy(prob.u0)
        if SciMLBase.isinplace(prob)
            fu = similar(u)
            prob.f(fu, u, prob.p)
        else
            fu = prob.f(u, prob.p)
        end

        ls_cache = init(prob, alg, fu, u)
        vjp_op = VecJacOperator(prob, fu, u; autodiff)

        alphas = Float64[]
        iter = 0
        for _ in 1:maxiters
            iter += 1
            svjp_op = StatefulJacobianOperator(vjp_op, u, prob.p)
            gs = svjp_op * fu .* 2
            δu = -gs

            ls_sol = solve!(ls_cache, u, δu)

            push!(alphas, ls_sol.step_size)
            @. u = u + ls_sol.step_size * δu
            gnorm = norm(gs)

            if SciMLBase.isinplace(prob)
                fu = similar(u)
                prob.f(fu, u, prob.p)
            else
                fu = prob.f(u, prob.p)
            end

            LineSearch.callback_into_cache!(ls_cache, fu)

            gnorm < g_atol && break
        end

        return fu, u, iter, alphas
    end

    @testset "LineSearches.jl with AutoEnzyme" begin
        @testset "OOP Problem" begin
            nlf(x, p) = [p[1] - x[1], 10.0 * (x[2] - x[1]^2)]
            nlp = NonlinearProblem(nlf, [-1.0, 1.0], [1.0])

            autodiff = AutoEnzyme()
            @testset "method: $(nameof(typeof(method)))" for method in (
                LineSearches.BackTracking(; order = 3),
                StrongWolfe(),
                HagerZhang(),
                MoreThuente()
            )
                linesearch = LineSearchesJL(; method, autodiff)
                fu, u, iter, alphas = gradient_descent(nlp, linesearch; autodiff)

                @test fu≈[0.0, 0.0] atol=1e-2
                @test u≈[1.0, 1.0] atol=1e-2
                @test !all(isone, alphas)
            end
        end

        @testset "In-Place Problem" begin
            nlf(dx, x, p) = (dx .= [p[1] - x[1], 10.0 * (x[2] - x[1]^2)])
            nlp = NonlinearProblem(nlf, [-1.0, 1.0], [1.0])

            autodiff = AutoEnzyme()
            @testset "method: $(nameof(typeof(method)))" for method in (
                LineSearches.BackTracking(; order = 3),
                StrongWolfe(),
                HagerZhang(),
                MoreThuente()
            )
                linesearch = LineSearchesJL(; method, autodiff)
                fu, u, iter, alphas = gradient_descent(nlp, linesearch; autodiff)

                @test fu≈[0.0, 0.0] atol=1e-2
                @test u≈[1.0, 1.0] atol=1e-2
                @test !all(isone, alphas)
            end
        end
    end

    @testset "Native Line Search with AutoEnzyme" begin
        @testset "OOP Problem" begin
            nlf(x, p) = [p[1] - x[1], 10.0 * (x[2] - x[1]^2)]
            nlp = NonlinearProblem(nlf, [-1.0, 1.0], [1.0])

            autodiff = AutoEnzyme()
            @testset "method: $(nameof(typeof(method)))" for method in (
                LiFukushimaLineSearch(),
                NoLineSearch(0.001),
                BackTracking(; order = Val(3), autodiff),
                BackTracking(; order = Val(2), autodiff)
            )
                fu, u, iter, alphas = gradient_descent(nlp, method; autodiff)

                @test fu≈[0.0, 0.0] atol=1e-1
                @test u≈[1.0, 1.0] atol=1e-1
                @test !all(isone, alphas)
            end
        end

        @testset "In-Place Problem" begin
            nlf(dx, x, p) = (dx .= [p[1] - x[1], 10.0 * (x[2] - x[1]^2)])
            nlp = NonlinearProblem(nlf, [-1.0, 1.0], [1.0])

            autodiff = AutoEnzyme()
            @testset "method: $(nameof(typeof(method)))" for method in (
                LiFukushimaLineSearch(),
                NoLineSearch(0.001),
                BackTracking(; order = Val(3), autodiff),
                BackTracking(; order = Val(2), autodiff)
            )
                fu, u, iter, alphas = gradient_descent(nlp, method; autodiff)

                @test fu≈[0.0, 0.0] atol=1e-1
                @test u≈[1.0, 1.0] atol=1e-1
                @test !all(isone, alphas)
            end
        end
    end
end
