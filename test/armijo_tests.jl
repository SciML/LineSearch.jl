using LineSearch, SciMLBase, StaticArrays, Test, ForwardDiff
using ADTypes: AutoForwardDiff
using JLArrays: JLArray
using LinearAlgebra: dot

@testset "Geometric and projected Armijo" begin
    @testset "Input validation" begin
        for Alg in (ArmijoLineSearch, ProjectedBackTracking)
            for kwargs in (
                    (; c_1 = 0), (; c_1 = 1), (; contraction = 0), (; contraction = 1),
                    (; initial_alpha = Inf), (; initial_alpha = 0), (; maxiters = 0),
                )
                @test_throws ArgumentError Alg(; kwargs...)
            end
        end
        prob = NonlinearProblem((u, p) -> u, [1.0])
        @test_throws DimensionMismatch init(prob, ProjectedBackTracking(), [1.0], [1.0]; lb = [0.0, 0.0])
        @test_throws ArgumentError init(prob, ProjectedBackTracking(), [1.0], [1.0]; lb = 2.0)
        @test_throws ArgumentError init(prob, ProjectedBackTracking(), [1.0], [1.0]; lb = 2.0, ub = 0.0)
    end

    @testset "Residual storage: $inplace / $shape" for inplace in (false, true), shape in (:vector, :static, :scalar, :matrix)
        inplace && shape in (:scalar, :static) && continue
        u = shape === :scalar ? 1.0 : shape === :static ? SVector(1.0, 1.0) : shape === :matrix ? ones(1, 2) : ones(2)
        f(u, p) = u .* p
        f!(out, u, p) = (out .= u .* p)
        nf = inplace ? NonlinearFunction{true}(f!; jvp = (out, v, u, p) -> (out .= p .* v)) :
            NonlinearFunction{false}(f; jvp = (v, u, p) -> p .* v)
        prob = NonlinearProblem(nf, u, 1.0)
        for alg in (ArmijoLineSearch(), ProjectedBackTracking())
            stats = SciMLBase.NLStats(0, 0, 0, 0, 0)
            cache = init(prob, alg, f(u, 1.0), u; lb = 0.0, ub = 2.0, stats)
            sol = solve!(cache, u, -4 .* u; gradient = u, ϕ0 = dot(u, u) / 2)
            @test SciMLBase.successful_retcode(sol.retcode)
            expected_alpha = alg isa ArmijoLineSearch ? 0.25 : 1.0
            @test sol.step_size == expected_alpha
            trial = get_trial(cache)
            @test trial.u == zero(u)
            @test trial.fu == zero(u)
            @test typeof(trial.u) == typeof(u)
            @test stats.nf == (alg isa ArmijoLineSearch ? 3 : 1)
            @test u == one.(u)
            set_initial_step!(cache, 0.125)
            reinit!(cache; p = 2.0)
            sol = solve!(cache, u, -4 .* u; gradient = 4 .* u, ϕ0 = 2 * dot(u, u))
            @test sol.step_size == expected_alpha
            @test get_trial(cache).fu == f(get_trial(cache).u, 2.0)
            @test_throws ArgumentError set_initial_step!(cache, -1)
        end
    end

    @testset "Projection arc and residual reuse" begin
        seen = Vector{Float64}[]
        f(u, p) = (push!(seen, copy(u)); @assert all(0 .<= u .<= [0.001, 1]); u .- p)
        u, p = zeros(2), ones(2)
        prob = NonlinearLeastSquaresProblem(f, u, p; lb = 0.0, ub = [0.001, 1.0])
        cache = init(prob, ProjectedBackTracking(c_1 = 0.5), -p, u)
        sol = solve!(cache, u, [100.0, 1.0]; gradient = -p, ϕ0 = 1.0)
        @test sol.step_size == 1
        @test SciMLBase.successful_retcode(sol.retcode)
        @test get_trial(cache).u == [0.001, 1.0]
        @test length(seen) == 1
        @test sol.ϕ == dot(get_trial(cache).fu, get_trial(cache).fu) / 2
        @test_throws ArgumentError solve!(cache, u, ones(2))
        @test_throws ArgumentError solve!(cache, [-1.0, 0.0], ones(2); gradient = -p)
    end

    @testset "Projection can uncover descent after contraction" begin
        u, p = zeros(2), [1.0, -1.0]
        prob = NonlinearProblem((u, p) -> u .- p, u, p; lb = 0.0, ub = [0.01, Inf])
        cache = init(prob, ProjectedBackTracking(), -p, u)
        sol = solve!(cache, u, [2.0, 1.0]; gradient = -p, ϕ0 = 1.0)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test sol.step_size == 1 / 128
        @test get_trial(cache).u == [0.01, 1 / 128]
    end

    @testset "Floating-point step type" begin
        for T in (Float32, Float64), alg in (ArmijoLineSearch(), ProjectedBackTracking())
            u = T[1]
            prob = NonlinearProblem((u, p) -> u, u)
            cache = init(prob, alg, u, u; need_deriv = false)
            sol = solve!(cache, u, -4u; gradient = u, ϕ0 = T(0.5))
            @test sol.step_size === T(0.25)
        end
    end

    @testset "Fixed and one-sided bounds" begin
        u = [1.0, 0.0, 0.0]
        prob = NonlinearProblem(
            (u, p) -> u .- p, u, [1.0, 2.0, -2.0];
            lb = [1.0, 0.0, -Inf], ub = [1.0, Inf, 0.0]
        )
        cache = init(prob, ProjectedBackTracking(), [0.0, -2.0, 2.0], u)
        sol = solve!(cache, u, [100.0, 2.0, -2.0]; gradient = [0.0, -2.0, 2.0])
        @test SciMLBase.successful_retcode(sol.retcode)
        @test get_trial(cache).u == [1.0, 2.0, -2.0]
    end

    @testset "Objective merit and supplied derivatives" begin
        f(u, p) = sum(abs2, u) / 2
        of = OptimizationFunction(f; grad = (g, u, p) -> copyto!(g, u))
        prob = OptimizationProblem(of, [1.0])
        for alg in (ArmijoLineSearch(), ProjectedBackTracking())
            cache = init(prob, alg, prob.u0)
            sol = solve!(cache, prob.u0, [-4.0]; gradient = [1.0])
            @test sol.step_size == 0.25
            @test get_trial(cache) == (; u = [0.0], fu = nothing)
        end
        cache = init(prob, ArmijoLineSearch(), prob.u0)
        @test solve!(cache, prob.u0, [-4.0]).step_size == 0.25
        value_only = OptimizationProblem((u, p) -> sum(abs2, u) / 2, [1.0])
        cache = init(value_only, ArmijoLineSearch(), [1.0]; need_deriv = false)
        @test solve!(cache, [1.0], [-4.0]; ϕ0 = 0.5, dϕ0 = -4.0).step_size == 0.25
        nf = NonlinearFunction{false}((u, p) -> u; jvp = (v, u, p) -> v)
        prob = NonlinearProblem(nf, [1.0])
        cache = init(prob, ArmijoLineSearch(), [1.0], [1.0])
        @test solve!(cache, [1.0], [-4.0]).step_size == 0.25
    end

    @testset "GPUArrays without scalar indexing" begin
        u = JLArray([1.0, 1.0])
        f!(out, u, p) = (out .= u)
        prob = NonlinearProblem(f!, u)
        for alg in (ArmijoLineSearch(), ProjectedBackTracking())
            cache = init(prob, alg, u, u; need_deriv = false, lb = 0.0, ub = 2.0)
            sol = solve!(cache, u, -4u; gradient = u, ϕ0 = 1.0)
            @test SciMLBase.successful_retcode(sol.retcode)
            @test Array(get_trial(cache).u) == zeros(2)
            @test Array(get_trial(cache).fu) == zeros(2)
        end
    end

    @testset "Algorithm derivative backend" begin
        prob = NonlinearProblem((u, p) -> u, [1.0])
        cache = init(prob, ArmijoLineSearch(; autodiff = AutoForwardDiff()), [1.0], [1.0]; autodiff = nothing)
        @test solve!(cache, [1.0], [-4.0]).step_size == 0.25
    end

    @testset "Rejection, nonfinite values, and evaluation limits" begin
        calls = Ref(0)
        f(u, p) = (calls[] += 1; u[1] <= 0 ? [NaN] : u)
        prob = NonlinearProblem(f, [1.0])
        for alg in (ArmijoLineSearch(maxiters = 4), ProjectedBackTracking(maxiters = 4))
            cache = init(prob, alg, [1.0], [1.0]; need_deriv = false)
            calls[] = 0
            sol = solve!(cache, [1.0], [-4.0]; gradient = [1.0], ϕ0 = 0.5)
            @test sol.step_size == 0.125
            @test SciMLBase.successful_retcode(sol.retcode)
            @test calls[] == 4
            calls[] = 0
            sol = solve!(cache, [1.0], [1.0]; gradient = [1.0], ϕ0 = 0.5)
            @test !SciMLBase.successful_retcode(sol.retcode)
            @test sol.step_size == 0
            @test calls[] == 0
        end
        for Alg in (ArmijoLineSearch, ProjectedBackTracking)
            cache = init(prob, Alg(maxiters = 3), [1.0], [1.0]; need_deriv = false)
            calls[] = 0
            sol = solve!(cache, [1.0], [-4.0]; gradient = [1.0], ϕ0 = 0.5)
            @test !SciMLBase.successful_retcode(sol.retcode)
            @test calls[] == 3
        end
    end
end
