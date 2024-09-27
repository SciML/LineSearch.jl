# Test based on https://julianlsolvers.github.io/LineSearches.jl/stable/examples/generated/customoptimizer.html
@testsetup module CustomOptimizer
using LinearAlgebra, SciMLBase, LineSearch, SciMLJacobianOperators

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

export gradient_descent

end

@testitem "LineSearches.jl: Custom Optimizer" setup=[CustomOptimizer] begin
    using LineSearches, SciMLBase
    using ADTypes, Tracker, ForwardDiff, Zygote, Enzyme, ReverseDiff, FiniteDiff

    @testset "OOP Problem" begin
        nlf(x, p) = [p[1] - x[1], 10.0 * (x[2] - x[1]^2)]
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [1.0])

        @testset for autodiff in (AutoTracker(), AutoForwardDiff(), AutoZygote(),
            AutoEnzyme(), AutoReverseDiff(), AutoFiniteDiff()
        )
            @testset "method: $(nameof(typeof(method)))" for method in (
                BackTracking(; order = 3), StrongWolfe(),
                HagerZhang(), MoreThuente()
            )
                linesearch = LineSearchesJL(; method, autodiff)
                fu, u, iter, alphas = gradient_descent(nlp, linesearch; autodiff)

                @test fu≈[0.0, 0.0] atol=1e-2
                @test u≈[1.0, 1.0] atol=1e-2
                @test !all(isone, alphas)
            end
        end
    end

    @testset "In-Place Problem" begin
        nlf(dx, x, p) = (dx .= [p[1] - x[1], 10.0 * (x[2] - x[1]^2)])
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [1.0])

        @testset for autodiff in (
            AutoForwardDiff(), AutoEnzyme(), AutoReverseDiff(), AutoFiniteDiff()
        )
            @testset "method: $(nameof(typeof(method)))" for method in (
                BackTracking(; order = 3), StrongWolfe(),
                HagerZhang(), MoreThuente()
            )
                linesearch = LineSearchesJL(; method, autodiff)
                fu, u, iter, alphas = gradient_descent(nlp, linesearch; autodiff)

                @test fu≈[0.0, 0.0] atol=1e-2
                @test u≈[1.0, 1.0] atol=1e-2
                @test !all(isone, alphas)
            end
        end
    end
end

@testitem "Native Line Search: Custom Optimizer" setup=[CustomOptimizer] begin
    using LineSearches, SciMLBase
    using ADTypes, Tracker, ForwardDiff, Zygote, Enzyme, ReverseDiff, FiniteDiff

    @testset "OOP Problem" begin
        nlf(x, p) = [p[1] - x[1], 10.0 * (x[2] - x[1]^2)]
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [1.0])

        @testset for autodiff in (AutoTracker(), AutoForwardDiff(), AutoZygote(),
            AutoEnzyme(), AutoReverseDiff(), AutoFiniteDiff()
        )
            @testset "method: $(nameof(typeof(method)))" for method in (
                LiFukushimaLineSearch(),
                NoLineSearch(0.001)
            )
                fu, u, iter, alphas = gradient_descent(nlp, method; autodiff)

                @test fu≈[0.0, 0.0] atol=1e-1
                @test u≈[1.0, 1.0] atol=1e-1
                @test !all(isone, alphas)
            end
        end
    end

    @testset "In-Place Problem" begin
        nlf(dx, x, p) = (dx .= [p[1] - x[1], 10.0 * (x[2] - x[1]^2)])
        nlp = NonlinearProblem(nlf, [-1.0, 1.0], [1.0])

        @testset for autodiff in (
            AutoForwardDiff(), AutoEnzyme(), AutoReverseDiff(), AutoFiniteDiff()
        )
            @testset "method: $(nameof(typeof(method)))" for method in (
                LiFukushimaLineSearch(),
                NoLineSearch(0.001)
            )
                fu, u, iter, alphas = gradient_descent(nlp, method; autodiff)

                @test fu≈[0.0, 0.0] atol=1e-1
                @test u≈[1.0, 1.0] atol=1e-1
                @test !all(isone, alphas)
            end
        end
    end
end
