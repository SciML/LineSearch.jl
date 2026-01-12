# Allocation tests to prevent performance regressions
# These tests verify that critical IIP (in-place) code paths remain zero-allocation
# Note: Allocation behavior varies across Julia versions, so tests are skipped on older versions

@testitem "Allocation Tests: LiFukushimaLineSearch IIP" tags = [:alloc] begin
    using LineSearch, SciMLBase, CommonSolve, Test

    # Skip allocation tests on Julia < 1.11 where allocation behavior differs
    if VERSION < v"1.12"
        @test_skip "Skipped on Julia $(VERSION) - allocation tests require Julia 1.12+"
    else
        function nlf_iip!(dx, x, p)
            dx[1] = x[1]^2 - p[1]
            dx[2] = x[2]^2 - p[2]
            return nothing
        end

        nlp = NonlinearProblem(nlf_iip!, [-1.0, 1.0], [3.0, 3.0])
        fu = similar(nlp.u0)
        nlf_iip!(fu, nlp.u0, nlp.p)
        u = copy(nlp.u0)
        du = [-0.5, 0.5]

        alg = LiFukushimaLineSearch()
        cache = init(nlp, alg, fu, u)

        # Warmup
        for _ in 1:10
            solve!(cache, u, du)
        end

        @testset "solve! is zero-allocation" begin
            allocs = @allocated solve!(cache, u, du)
            @test allocs == 0
        end
    end
end

@testitem "Allocation Tests: RobustNonMonotoneLineSearch IIP" tags = [:alloc] begin
    using LineSearch, SciMLBase, CommonSolve, Test

    # Skip allocation tests on Julia < 1.11 where allocation behavior differs
    if VERSION < v"1.12"
        @test_skip "Skipped on Julia $(VERSION) - allocation tests require Julia 1.12+"
    else
        function nlf_iip!(dx, x, p)
            dx[1] = x[1]^2 - p[1]
            dx[2] = x[2]^2 - p[2]
            return nothing
        end

        nlp = NonlinearProblem(nlf_iip!, [-1.0, 1.0], [3.0, 3.0])
        fu = similar(nlp.u0)
        nlf_iip!(fu, nlp.u0, nlp.p)
        u = copy(nlp.u0)
        du = [-0.5, 0.5]

        alg = RobustNonMonotoneLineSearch()
        cache = init(nlp, alg, fu, u)

        # Warmup
        for _ in 1:10
            solve!(cache, u, du)
        end

        @testset "solve! is zero-allocation" begin
            allocs = @allocated solve!(cache, u, du)
            @test allocs == 0
        end

        @testset "callback_into_cache! is zero-allocation" begin
            allocs = @allocated LineSearch.callback_into_cache!(cache, fu)
            @test allocs == 0
        end
    end
end

@testitem "Allocation Tests: NoLineSearch" tags = [:alloc] begin
    using LineSearch, SciMLBase, CommonSolve, Test

    # Skip allocation tests on Julia < 1.11 where allocation behavior differs
    if VERSION < v"1.12"
        @test_skip "Skipped on Julia $(VERSION) - allocation tests require Julia 1.12+"
    else
        function nlf_iip!(dx, x, p)
            dx[1] = x[1]^2 - p[1]
            dx[2] = x[2]^2 - p[2]
            return nothing
        end

        nlp = NonlinearProblem(nlf_iip!, [-1.0, 1.0], [3.0, 3.0])
        fu = similar(nlp.u0)
        nlf_iip!(fu, nlp.u0, nlp.p)
        u = copy(nlp.u0)
        du = [-0.5, 0.5]

        alg = NoLineSearch(1.0)
        cache = init(nlp, alg, fu, u)

        # Warmup
        for _ in 1:10
            solve!(cache, u, du)
        end

        @testset "solve! is zero-allocation" begin
            allocs = @allocated solve!(cache, u, du)
            @test allocs == 0
        end
    end
end

@testitem "Allocation Tests: StaticLiFukushimaLineSearch" tags = [:alloc] begin
    using LineSearch, SciMLBase, CommonSolve, StaticArrays, Test

    # Skip allocation tests on Julia < 1.11 where allocation behavior differs
    if VERSION < v"1.12"
        @test_skip "Skipped on Julia $(VERSION) - allocation tests require Julia 1.12+"
    else
        static_f(u::SVector{2}, p) = SVector(u[1]^2 - p[1], u[2]^2 - p[2])

        prob = NonlinearProblem(static_f, SVector(-1.0, 1.0), SVector(3.0, 3.0))
        fu = static_f(prob.u0, prob.p)
        u = prob.u0
        du = SVector(-0.5, 0.5)

        # Use nan_maxiters=nothing to get the static (non-allocating) path
        alg = LiFukushimaLineSearch(; nan_maxiters = nothing)
        cache = init(prob, alg, fu, u)

        # Warmup
        for _ in 1:10
            solve!(cache, u, du)
        end

        @testset "solve! is zero-allocation" begin
            allocs = @allocated solve!(cache, u, du)
            @test allocs == 0
        end
    end
end

@testitem "Allocation Tests: Scalar LiFukushimaLineSearch" tags = [:alloc] begin
    using LineSearch, SciMLBase, CommonSolve, Test

    # Skip allocation tests on Julia < 1.11 where allocation behavior differs
    if VERSION < v"1.12"
        @test_skip "Skipped on Julia $(VERSION) - allocation tests require Julia 1.12+"
    else
        scalar_f(u, p) = u^2 - p

        prob = NonlinearProblem(scalar_f, 2.0, 4.0)
        fu = scalar_f(prob.u0, prob.p)
        u = prob.u0
        du = -0.5

        # Use nan_maxiters=nothing to get the static (non-allocating) path
        alg = LiFukushimaLineSearch(; nan_maxiters = nothing)
        cache = init(prob, alg, fu, u)

        # Warmup
        for _ in 1:10
            solve!(cache, u, du)
        end

        @testset "solve! is zero-allocation" begin
            allocs = @allocated solve!(cache, u, du)
            @test allocs == 0
        end
    end
end

@testitem "Allocation Tests: Larger Problems (10D)" tags = [:alloc] begin
    using LineSearch, SciMLBase, CommonSolve, Test

    # Skip allocation tests on Julia < 1.11 where allocation behavior differs
    if VERSION < v"1.12"
        @test_skip "Skipped on Julia $(VERSION) - allocation tests require Julia 1.12+"
    else
        function nlf_10d!(dx, x, p)
            for i in 1:10
                dx[i] = x[i]^2 - p[i]
            end
            return nothing
        end

        nlp = NonlinearProblem(nlf_10d!, ones(10), 3 * ones(10))
        fu = similar(nlp.u0)
        nlf_10d!(fu, nlp.u0, nlp.p)
        u = copy(nlp.u0)
        du = -0.5 * ones(10)

        @testset "LiFukushimaLineSearch 10D" begin
            alg = LiFukushimaLineSearch()
            cache = init(nlp, alg, fu, u)
            for _ in 1:10
                solve!(cache, u, du)
            end
            allocs = @allocated solve!(cache, u, du)
            @test allocs == 0
        end

        @testset "RobustNonMonotoneLineSearch 10D" begin
            alg = RobustNonMonotoneLineSearch()
            cache = init(nlp, alg, fu, u)
            for _ in 1:10
                solve!(cache, u, du)
            end
            allocs = @allocated solve!(cache, u, du)
            @test allocs == 0
        end
    end
end
