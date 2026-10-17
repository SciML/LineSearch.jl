# AllocCheck static analysis of the native `solve!` inner loop.
#
# Runtime `@allocated` checks live in Core (`gpu_arrays_tests.jl`). AllocCheck is
# stricter LLVM IR analysis: the Vector/`@bb` broadcast path still reports
# potential sites even when steady-state runtime allocation is zero, so this
# lane pins the heap-free StaticArrays / Number residual path that GPU kernels
# and static solvers actually compile.
using AllocCheck
using CommonSolve
using LineSearch
using SciMLBase
using StaticArrays
using Test

@testset "AllocCheck: solve! is allocation-free" begin
    Fres(u, p) = @. u^2 - 1
    jvp(v, u, p) = @. 2 * u * v
    residual_grad(u, fu) = @. 2 * u * fu

    ALGS = (
        "HagerZhang" => HagerZhangLineSearch(),
        "MoreThuente" => MoreThuenteLineSearch(),
        "BackTracking" => BackTracking(),
        "GoldenSection" => GoldenSection(),
        "LiFukushima" => LiFukushimaLineSearch(nan_maxiters = nothing),
        "StrongWolfe" => StrongWolfeLineSearch(),
        "RobustNonMonotone" => RobustNonMonotoneLineSearch(),
        "NoLineSearch" => NoLineSearch(),
    )

    @testset "SArray residual: $name" for (name, alg) in ALGS
        u = @SVector [1.5, -0.5, 2.0]
        fu = Fres(u, nothing)
        du = .-residual_grad(u, fu)
        nf = NonlinearFunction{false}(Fres; jvp)
        prob = NonlinearProblem(nf, u)
        kwargs = alg isa StrongWolfeLineSearch ?
            (; grad_f = (x, p) -> residual_grad(x, Fres(x, p))) : (;)
        cache = CommonSolve.init(prob, alg, fu, u; kwargs...)
        sol = CommonSolve.solve!(cache, u, du)
        @test sol.retcode == SciMLBase.ReturnCode.Success || alg isa NoLineSearch

        allocs = check_allocs(
            CommonSolve.solve!, (typeof(cache), typeof(u), typeof(du))
        )
        @test isempty(allocs)
        if !isempty(allocs)
            @info "AllocCheck sites for $name" allocs
        end
    end

    @testset "Number residual: $name" for (name, alg) in (
            "HagerZhang" => HagerZhangLineSearch(),
            "MoreThuente" => MoreThuenteLineSearch(),
            "BackTracking" => BackTracking(),
            "GoldenSection" => GoldenSection(),
            "LiFukushima" => LiFukushimaLineSearch(nan_maxiters = nothing),
            "StrongWolfe" => StrongWolfeLineSearch(),
            "NoLineSearch" => NoLineSearch(),
        )
        u = 1.5
        fu = u^2 - 1
        du = -2 * u * fu
        nf = NonlinearFunction{false}((x, p) -> x^2 - 1; jvp = (v, x, p) -> 2 * x * v)
        prob = NonlinearProblem(nf, u)
        kwargs = alg isa StrongWolfeLineSearch ?
            (; grad_f = (x, p) -> 2 * x * (x^2 - 1)) : (;)
        cache = CommonSolve.init(prob, alg, fu, u; kwargs...)
        sol = CommonSolve.solve!(cache, u, du)
        @test sol.retcode == SciMLBase.ReturnCode.Success || alg isa NoLineSearch

        allocs = check_allocs(
            CommonSolve.solve!, (typeof(cache), typeof(u), typeof(du))
        )
        @test isempty(allocs)
    end
end
