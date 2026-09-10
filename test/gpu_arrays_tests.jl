# Steady-state allocation contracts and GPUArrays compatibility (via JLArrays).
#
# The merit-driven native searches allocate only in `init`. After that, `solve!`
# reuses `u_cache`/`fu_cache`/`jv_cache` and must not grow allocations. JLArrays
# stand in for CUDA/ROC device arrays: the inner loop must not scalar-index.
using LineSearch, Test
using SciMLBase, CommonSolve, LinearAlgebra
using SciMLBase: ReturnCode, NonlinearProblem, NonlinearFunction,
    OptimizationProblem, OptimizationFunction
using JLArrays
using StaticArrays
using ADTypes: AutoForwardDiff

@testset "Inner-loop allocations and GPUArrays" begin

    # Broadcast-friendly residual F(u) = u.^2 .- 1  (no scalar indexing).
    Fres!(fu, u, p) = (@. fu = u^2 - 1; nothing)
    Fres(u, p) = @. u^2 - 1
    jvp!(Jv, v, u, p) = (@. Jv = 2 * u * v; nothing)
    jvp(v, u, p) = @. 2 * u * v
    # Array path is a tight loop so solve! allocation tests stay at zero; the
    # generic path uses broadcast reductions so JLArray does not scalar-index.
    function obj(u::Array, p)
        s = zero(real(eltype(u)))
        @inbounds for i in eachindex(u)
            s += abs2(u[i]^2 - 1)
        end
        return s / 2
    end
    obj(u, p) = sum((u .^ 2 .- 1) .^ 2) / 2
    obj_grad!(G, u, p) = (@. G = 2 * u * (u^2 - 1); G)

    function residual_direction(u, fu)
        g = similar(u)
        @. g = 2 * u * fu
        return .-g
    end

    MERIT_ALGS = (
        "HagerZhang" => HagerZhangLineSearch(),
        "MoreThuente" => MoreThuenteLineSearch(),
        "StrongWolfe" => StrongWolfeLineSearch(),
        "BackTracking" => BackTracking(),
    )
    ALL_NATIVE = (
        MERIT_ALGS...,
        "GoldenSection" => GoldenSection(),
        "LiFukushima" => LiFukushimaLineSearch(),
        "RobustNonMonotone" => RobustNonMonotoneLineSearch(),
        "NoLineSearch" => NoLineSearch(),
    )

    function assert_steady_allocs(cache, u, du; max_bytes = 0)
        CommonSolve.solve!(cache, u, du)   # warm-up / compile
        a1 = @allocated CommonSolve.solve!(cache, u, du)
        a2 = @allocated CommonSolve.solve!(cache, u, du)
        @test a1 == a2
        if VERSION ≥ v"1.11"
            @test a2 ≤ max_bytes
        else
            @test a2 ≤ max(max_bytes, 256)
        end
        return a2
    end

    # ------------------------------------------------ objective merit Vector

    @testset "objective merit solve! is non-allocating: $name" for (name, alg) in MERIT_ALGS
        uh = [1.5, -0.5, 2.0]
        gh = zeros(3)
        obj_grad!(gh, uh, nothing)
        du = .-gh
        prob = OptimizationProblem(OptimizationFunction(obj; grad = obj_grad!), uh)
        cache = CommonSolve.init(prob, alg, uh)
        sol = CommonSolve.solve!(cache, uh, du)
        @test sol.retcode == ReturnCode.Success
        @test sol.step_size > 0
        assert_steady_allocs(cache, uh, du)
    end

    @testset "objective GoldenSection is non-allocating" begin
        uh = [1.5, -0.5, 2.0]
        du = .-uh
        prob = OptimizationProblem(OptimizationFunction(obj), uh)
        cache = CommonSolve.init(prob, GoldenSection(), uh)
        assert_steady_allocs(cache, uh, du)
    end

    # ------------------------------------ residual merit IIP + analytic JVP

    @testset "IIP residual + analytic jvp is non-allocating: $name" for (name, alg) in ALL_NATIVE
        u = [1.5, -0.5, 2.0]
        fu = similar(u)
        Fres!(fu, u, nothing)
        du = residual_direction(u, fu)
        nf = NonlinearFunction{true}(Fres!; jvp = jvp!)
        prob = NonlinearProblem(nf, u)
        cache = CommonSolve.init(prob, alg, fu, u)
        sol = CommonSolve.solve!(cache, u, du)
        @test sol.retcode == ReturnCode.Success || alg isa NoLineSearch
        # Derivative-free searches never touch jv_cache; the rest must stay at 0
        # once the JVP writes into the preallocated buffer.
        assert_steady_allocs(cache, u, du)
    end

    # ----------------------------------------------- residual merit + VJP

    @testset "IIP residual + analytic vjp is non-allocating: $name" for (name, alg) in MERIT_ALGS
        u = [1.5, -0.5, 2.0]
        fu = similar(u)
        Fres!(fu, u, nothing)
        du = residual_direction(u, fu)
        # J is diagonal with entries 2u, so J' = J.
        vjp!(vJ, v, u, p) = (@. vJ = 2 * u * v; nothing)
        nf = NonlinearFunction{true}(Fres!; vjp = vjp!)
        prob = NonlinearProblem(nf, u)
        cache = CommonSolve.init(prob, alg, fu, u)
        sol = CommonSolve.solve!(cache, u, du)
        @test sol.retcode == ReturnCode.Success
        @test cache.merit_eval.jv_cache !== nothing
        @test length(cache.merit_eval.jv_cache) == length(u)
        assert_steady_allocs(cache, u, du)
    end

    # ------------------------------------- ϕ0/dϕ0 kwargs avoid re-eval at 0

    @testset "ϕ0/dϕ0 kwargs keep residual solve! non-allocating: $name" for (name, alg) in MERIT_ALGS
        u = [1.5, -0.5, 2.0]
        fu = similar(u)
        Fres!(fu, u, nothing)
        du = residual_direction(u, fu)
        nf = NonlinearFunction{true}(Fres!; jvp = jvp!)
        prob = NonlinearProblem(nf, u)
        cache = CommonSolve.init(prob, alg, fu, u)
        ϕ0 = sum(abs2, fu) / 2
        dϕ0 = dot(fu, (@. 2 * u * du))
        CommonSolve.solve!(cache, u, du; ϕ0, dϕ0)
        a1 = @allocated CommonSolve.solve!(cache, u, du; ϕ0, dϕ0)
        a2 = @allocated CommonSolve.solve!(cache, u, du; ϕ0, dϕ0)
        @test a1 == a2
        if VERSION ≥ v"1.11"
            @test a2 == 0
        else
            @test a2 ≤ 256
        end
    end

    # ---------------------------------------------------- StaticArrays path

    @testset "StaticArrays residual solve! is non-allocating: $name" for (name, alg) in (
            "HagerZhang" => HagerZhangLineSearch(),
            "MoreThuente" => MoreThuenteLineSearch(),
            "BackTracking" => BackTracking(),
            "GoldenSection" => GoldenSection(),
            "LiFukushima" => LiFukushimaLineSearch(nan_maxiters = nothing),
            "StrongWolfe" => StrongWolfeLineSearch(),
        )
        u = @SVector [1.5, -0.5, 2.0]
        fu = Fres(u, nothing)
        du = .-(@. 2 * u * fu)
        nf = NonlinearFunction{false}(Fres; jvp)
        prob = NonlinearProblem(nf, u)
        kwargs = alg isa StrongWolfeLineSearch ?
            (; grad_f = (x, p) -> (@. 2 * x * (x^2 - 1))) : (;)
        cache = CommonSolve.init(prob, alg, fu, u; kwargs...)
        sol = CommonSolve.solve!(cache, u, du)
        @test sol.retcode == ReturnCode.Success
        assert_steady_allocs(cache, u, du)
    end

    # ---------------------------------------------- JLArrays (GPUArrays stand-in)

    @testset "JLArray residual IIP works without scalar indexing: $name" for (name, alg) in ALL_NATIVE
        uh = [1.5, -0.5, 2.0]
        fuh = similar(uh)
        Fres!(fuh, uh, nothing)
        duh = Array(residual_direction(uh, fuh))
        u = jl(uh)
        fu = jl(fuh)
        du = jl(duh)
        nf = NonlinearFunction{true}(Fres!; jvp = jvp!)
        prob = NonlinearProblem(nf, u)

        cache = CommonSolve.init(prob, alg, fu, u)
        sol = CommonSolve.solve!(cache, u, du)
        @test sol.retcode == ReturnCode.Success || alg isa NoLineSearch
        @test sol.step_size isa Real
        @test Float64(sol.step_size) ≥ 0

        # Match the host result for the merit-driven searches.
        if alg isa Union{
                HagerZhangLineSearch, MoreThuenteLineSearch,
                StrongWolfeLineSearch, BackTracking,
            }
            host = CommonSolve.solve!(
                CommonSolve.init(
                    NonlinearProblem(NonlinearFunction{true}(Fres!; jvp = jvp!), uh),
                    alg, fuh, uh
                ),
                uh, duh
            )
            @test Float64(sol.step_size) ≈ Float64(host.step_size) rtol = 1.0e-10
        end

        # Steady-state allocation is constant (JLArray kernels allocate a
        # bounded staging cost; it must not grow across calls).
        CommonSolve.solve!(cache, u, du)
        a1 = @allocated CommonSolve.solve!(cache, u, du)
        a2 = @allocated CommonSolve.solve!(cache, u, du)
        @test a1 == a2 || abs(Int(a1) - Int(a2)) ≤ 512
    end

    @testset "JLArray objective merit works without scalar indexing: $name" for (name, alg) in (
            MERIT_ALGS..., "GoldenSection" =>
                GoldenSection(),
        )
        uh = [1.5, -0.5, 2.0]
        gh = zeros(3)
        obj_grad!(gh, uh, nothing)
        u = jl(uh)
        du = jl(.-gh)
        optf = if alg isa GoldenSection
            OptimizationFunction(obj)
        else
            OptimizationFunction(obj; grad = obj_grad!)
        end
        prob = OptimizationProblem(optf, u)
        cache = CommonSolve.init(prob, alg, u)
        sol = CommonSolve.solve!(cache, u, du)
        @test sol.retcode == ReturnCode.Success
        @test Float64(sol.step_size) > 0

        host_du = .-gh
        host = CommonSolve.solve!(
            CommonSolve.init(OptimizationProblem(optf, uh), alg, uh), uh, host_du
        )
        @test Float64(sol.step_size) ≈ Float64(host.step_size) rtol = 1.0e-10
    end

    # --------------------------------------- accepted-point caches stay on-device

    @testset "JLArray accepted-point caches keep array type" begin
        uh = [1.5, -0.5, 2.0]
        gh = zeros(3)
        obj_grad!(gh, uh, nothing)
        u = jl(uh)
        du = jl(.-gh)
        prob = OptimizationProblem(OptimizationFunction(obj; grad = obj_grad!), u)
        cache = CommonSolve.init(prob, HagerZhangLineSearch(), u)
        sol = CommonSolve.solve!(cache, u, du)
        @test sol.retcode == ReturnCode.Success
        @test cache.merit_eval.u_cache isa JLArray
        @test cache.merit_eval.fu_cache isa JLArray
    end

    @testset "JLArray residual jv_cache stays on-device" begin
        uh = [1.5, -0.5, 2.0]
        fuh = similar(uh)
        Fres!(fuh, uh, nothing)
        u = jl(uh)
        fu = jl(fuh)
        du = jl(Array(residual_direction(uh, fuh)))
        nf = NonlinearFunction{true}(Fres!; jvp = jvp!)
        cache = CommonSolve.init(NonlinearProblem(nf, u), MoreThuenteLineSearch(), fu, u)
        CommonSolve.solve!(cache, u, du)
        @test cache.merit_eval.jv_cache isa JLArray
        @test cache.merit_eval.u_cache isa JLArray
        @test cache.merit_eval.fu_cache isa JLArray
    end

    # ----------------------------------------------- set_initial_step! + reuse

    @testset "repeated solve! with set_initial_step! stays non-allocating" begin
        u = [1.5, -0.5, 2.0]
        fu = similar(u)
        Fres!(fu, u, nothing)
        du = residual_direction(u, fu)
        nf = NonlinearFunction{true}(Fres!; jvp = jvp!)
        prob = NonlinearProblem(nf, u)
        for alg in (HagerZhangLineSearch(), MoreThuenteLineSearch(), BackTracking())
            cache = CommonSolve.init(prob, alg, fu, u)
            CommonSolve.solve!(cache, u, du)
            set_initial_step!(cache, 0.5)
            CommonSolve.solve!(cache, u, du)   # warm the new α_init path
            set_initial_step!(cache, 0.75)
            a1 = @allocated CommonSolve.solve!(cache, u, du)
            set_initial_step!(cache, 0.6)
            a2 = @allocated CommonSolve.solve!(cache, u, du)
            @test a1 == a2
            if VERSION ≥ v"1.11"
                @test a2 == 0
            end
        end
    end
end
