using Test, LineSearch, SciMLBase, ADTypes, Enzyme
using DifferentiationInterface, ForwardDiff, Tracker, Zygote, ReverseDiff, FiniteDiff
using SciMLJacobianOperators, LinearAlgebra, LineSearches

@info "Running Enzyme tests with Julia $(VERSION)"

# Test setup module for custom optimizer tests
module CustomOptimizer
    using LinearAlgebra, SciMLBase, LineSearch, SciMLJacobianOperators

    function gradient_descent(
            prob, alg; g_atol::Real = 1.0e-5, maxiters::Int = 10000, autodiff = nothing
        )
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

    export gradient_descent
end

using .CustomOptimizer

# Test setup module for root finding tests
module RootFinding
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

using .RootFinding

const OOP_AUTODIFFS = (AutoEnzyme(),)
const IIP_AUTODIFFS = (AutoEnzyme(),)

@testset "Enzyme: LineSearches.jl Custom Optimizer" begin
    @testset "OOP Problem" begin
        nlf(x, p) = [p[1] - x[1], 10.0 * (x[2] - x[1]^2)]
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [1.0])

        @testset for autodiff in OOP_AUTODIFFS
            @testset "method: $(nameof(typeof(method)))" for method in (
                    LineSearches.BackTracking(; order = 3),
                    StrongWolfe(),
                    HagerZhang(),
                    MoreThuente(),
                )
                linesearch = LineSearchesJL(; method, autodiff)
                fu, u, iter, alphas = gradient_descent(nlp, linesearch; autodiff)

                @test fu ≈ [0.0, 0.0] atol = 1.0e-2
                @test u ≈ [1.0, 1.0] atol = 1.0e-2
                @test !all(isone, alphas)
            end
        end
    end

    @testset "In-Place Problem" begin
        nlf(dx, x, p) = (dx .= [p[1] - x[1], 10.0 * (x[2] - x[1]^2)])
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [1.0])

        @testset for autodiff in IIP_AUTODIFFS
            @testset "method: $(nameof(typeof(method)))" for method in (
                    LineSearches.BackTracking(; order = 3),
                    StrongWolfe(),
                    HagerZhang(),
                    MoreThuente(),
                )
                linesearch = LineSearchesJL(; method, autodiff)
                fu, u, iter, alphas = gradient_descent(nlp, linesearch; autodiff)

                @test fu ≈ [0.0, 0.0] atol = 1.0e-2
                @test u ≈ [1.0, 1.0] atol = 1.0e-2
                @test !all(isone, alphas)
            end
        end
    end
end

@testset "Enzyme: Native Line Search Custom Optimizer" begin
    @testset "OOP Problem" begin
        nlf(x, p) = [p[1] - x[1], 10.0 * (x[2] - x[1]^2)]
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [1.0])

        @testset for autodiff in OOP_AUTODIFFS
            @testset "method: $(nameof(typeof(method)))" for method in (
                    LineSearch.BackTracking(; order = Val(3), autodiff),
                    LineSearch.BackTracking(; order = Val(2), autodiff),
                )
                fu, u, iter, alphas = gradient_descent(nlp, method; autodiff)

                @test fu ≈ [0.0, 0.0] atol = 1.0e-1
                @test u ≈ [1.0, 1.0] atol = 1.0e-1
                @test !all(isone, alphas)
            end
        end
    end

    @testset "In-Place Problem" begin
        nlf(dx, x, p) = (dx .= [p[1] - x[1], 10.0 * (x[2] - x[1]^2)])
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [1.0])

        @testset for autodiff in IIP_AUTODIFFS
            @testset "method: $(nameof(typeof(method)))" for method in (
                    LineSearch.BackTracking(; order = Val(3), autodiff),
                    LineSearch.BackTracking(; order = Val(2), autodiff),
                )
                fu, u, iter, alphas = gradient_descent(nlp, method; autodiff)

                @test fu ≈ [0.0, 0.0] atol = 1.0e-1
                @test u ≈ [1.0, 1.0] atol = 1.0e-1
                @test !all(isone, alphas)
            end
        end
    end
end

@testset "Enzyme: LineSearches.jl Newton Raphson" begin
    @testset "OOP Problem" begin
        nlf(x, p) = x .^ 2 .- p
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

        @testset for autodiff in OOP_AUTODIFFS
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

        @testset for autodiff in IIP_AUTODIFFS
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

@testset "Enzyme: Native Line Search Newton Raphson" begin
    @testset "OOP Problem" begin
        nlf(x, p) = x .^ 2 .- p
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

        @testset for autodiff in OOP_AUTODIFFS
            @testset "method: $(nameof(typeof(method)))" for method in (
                    LineSearch.BackTracking(; order = Val(3), autodiff),
                    LineSearch.BackTracking(; order = Val(2), autodiff),
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

        @testset for autodiff in IIP_AUTODIFFS
            @testset "method: $(nameof(typeof(method)))" for method in (
                    LineSearch.BackTracking(; order = Val(3), autodiff),
                    LineSearch.BackTracking(; order = Val(2), autodiff),
                )
                converged, fu, u, iter, alphas = newton_raphson(nlp, method)

                @test fu ≈ [0.0, 0.0] atol = 1.0e-3
                @test abs.(u) ≈ sqrt.([3.0, 3.0]) atol = 1.0e-3
            end
        end
    end
end
