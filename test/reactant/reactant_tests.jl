using CommonSolve: init, solve!
using LineSearch, LinearAlgebra, Reactant, ReactantCore, SciMLBase, Test

cube(u, p) = u .^ 3 .- 2
cube!(fu, u, p) = (fu .= u .^ 3 .- 2; nothing)
trig(u, p) = sin.(u) .+ u ./ 2 .- p
linear(u, p) = u .- p

function residual(f, u, p)
    SciMLBase.isinplace(NonlinearProblem(f, u, p)) || return f(u, p)
    fu = similar(u)
    f(fu, u, p)
    return fu
end

traced_if_compiling(x) = ReactantCore.within_compile() ? ReactantCore.promote_to_traced(x) : x

# One line search after a recorded history entry (`nsteps = 1`, so `η` is finite and the
# unit step can be rejected).
function line_search(f, u, du, p, alg = RobustNonMonotoneLineSearch())
    prob = NonlinearProblem(f, u, p)
    fu = residual(f, u, p)
    cache = init(prob, alg, fu, u)
    LineSearch.callback_into_cache!(cache, fu)
    sol = solve!(cache, u, du)
    return sol.step_size, sol.retcode == ReturnCode.Success, cache.history
end

spectral_coefficient(Δu, Δf) = clamp_spectral(dot(Δu, Δu) / dot(Δu, Δf), Δf)
function clamp_spectral(σ, fu)
    in_bounds = (1.0e-10 ≤ abs(σ)) & (abs(σ) ≤ 1.0e10)
    return ifelse(in_bounds, σ, clamp(inv(norm(fu)), 1.0, 1.0e5))
end

# DF-SANE (La Cruz, Martínez, Raydan 2006) outer loop, as in NonlinearSolve's `DFSane`.
# `unit_step = true` takes the spectral step with `α = 1` and no line search.
function dfsane(f, u0, p, maxiters; unit_step = false, abstol = 1.0e-10)
    prob = NonlinearProblem(f, u0, p)
    u = u0
    fu = f(u, p)
    ls = init(prob, RobustNonMonotoneLineSearch(), fu, u)
    σ = clamp_spectral(dot(u, u) / dot(u, fu), fu)
    nsteps = traced_if_compiling(0)
    done = traced_if_compiling(false)
    ReactantCore.@trace track_numbers = false while (nsteps < maxiters) & !done
        du = -σ .* fu
        sol = solve!(ls, u, du)
        ls_failed = !unit_step & !SciMLBase.successful_retcode(sol.retcode)
        α = ifelse(unit_step, one(σ), ifelse(ls_failed, zero(σ), sol.step_size))
        u_new = u .+ α .* du
        fu_new = f(u_new, p)
        σ = spectral_coefficient(u_new .- u, fu_new .- fu)
        u = u_new
        fu = fu_new
        LineSearch.callback_into_cache!(ls, fu)
        done = (maximum(abs, fu) ≤ abstol) | ls_failed
        nsteps = nsteps + 1
    end
    return u, nsteps, maximum(abs, fu) ≤ abstol
end

# Expected steps and return codes are those of the host loop in LineSearch v0.1.19.
@testset "compiled line search matches the host" begin
    u_cube = [3.0, -2.0]
    u_trig = collect(range(-2.0, 2.0; length = 5))
    default = RobustNonMonotoneLineSearch()
    cases = [
        "unit step rejected" => (cube, u_cube, nothing, default, 0.1, true),
        "unit step rejected, in place" => (cube!, u_cube, nothing, default, 0.1, true),
        "unit step accepted" => (trig, u_trig, 0.3, default, 1.0, true),
        "maxiters reached" => (
            cube, u_cube, nothing, RobustNonMonotoneLineSearch(; maxiters = 1), 1.0, false,
        ),
        "maxiters reached, sigma_1 = 2" => (
            cube, u_cube, nothing, RobustNonMonotoneLineSearch(; maxiters = 1, sigma_1 = 2),
            2.0, false,
        ),
    ]
    @testset "$name" for (name, (f, u, p, alg, α_expected, ok_expected)) in cases
        du = -residual(f, u, p)
        α_host, ok_host, _ = line_search(f, u, du, p, alg)
        @test α_host == α_expected
        @test ok_host == ok_expected
        α_jit, ok_jit, _ = @jit line_search(
            f, Reactant.to_rarray(u), Reactant.to_rarray(du), p, alg
        )
        @test Float64(α_jit) ≈ α_expected
        @test Bool(ok_jit) == ok_expected
    end
end

# Merits whose `sqrt(dot(x, x))` underflows to zero or overflows to `Inf`. Expected steps,
# return codes and history are those of the host loop in LineSearch v0.1.19.
@testset "compiled line search with finite extreme merits" begin
    alg = RobustNonMonotoneLineSearch(; n_exp = 1, maxiters = 3)
    cases = [
        "Float64, small" => ([1.0e-200, 1.0e-200], [1.0e-199, 1.0e-199], 0.0, -0.1),
        "Float32, small" => (Float32[1.0f-30, 1.0f-30], Float32[1.0f-29, 1.0f-29], 0.0f0, -0.1f0),
        "Float64, large" => ([1.0e200, 1.0e200], [-1.0e200, -1.0e200], 0.0, 1.0),
    ]
    @testset "$name" for (name, (u, du, p, α_expected)) in cases
        history_expected = fill(sqrt(2one(eltype(u))) * u[1], 10)
        α_host, ok_host, history_host = line_search(linear, u, du, p, alg)
        @test α_host == α_expected
        @test ok_host
        @test history_host ≈ history_expected
        α_jit, ok_jit, history_jit = @jit line_search(
            linear, Reactant.to_rarray(u), Reactant.to_rarray(du), p, alg
        )
        @test eltype(u)(α_jit) ≈ α_expected
        @test Bool(ok_jit)
        @test Array(history_jit) ≈ history_expected
    end
end

# Expected values are those of the host loop in LineSearch v0.1.19.
@testset "compiled line search on a scalar residual: $name" for (name, alg, α_expected, ok_expected) in (
        ("unit step rejected", RobustNonMonotoneLineSearch(; n_exp = 1), 0.1, true),
        ("maxiters reached", RobustNonMonotoneLineSearch(; n_exp = 1, maxiters = 1), 1.0, false),
    )
    u, du, p = 3.0, -60.0, 0.0
    α_host, ok_host, history_host = line_search(linear, u, du, p, alg)
    @test α_host == α_expected
    @test ok_host == ok_expected
    α_jit, ok_jit, history_jit = @jit line_search(
        linear, Reactant.ConcreteRNumber(u), Reactant.ConcreteRNumber(du), p, alg
    )
    @test Float64(α_jit) == α_expected
    @test Bool(ok_jit) == ok_expected
    @test Array(history_jit) == history_host
end

# NonlinearSolve's DFSane checks the line search with `SciMLBase.successful_retcode`.
function checked_line_search(f, u, du, p, alg)
    prob = NonlinearProblem(f, u, p)
    fu = residual(f, u, p)
    cache = init(prob, alg, fu, u)
    LineSearch.callback_into_cache!(cache, fu)
    sol = solve!(cache, u, du)
    return sol, SciMLBase.successful_retcode(sol.retcode)
end

@testset "successful_retcode of the compiled return code: $name" for (name, alg, ok_expected) in (
        ("success", RobustNonMonotoneLineSearch(), true),
        ("failure", RobustNonMonotoneLineSearch(; maxiters = 1), false),
    )
    u = [3.0, -2.0]
    du = -cube(u, nothing)
    sol_host, ok_host = checked_line_search(cube, u, du, nothing, alg)
    @test ok_host == ok_expected
    sol_jit, ok_jit = @jit checked_line_search(
        cube, Reactant.to_rarray(u), Reactant.to_rarray(du), nothing, alg
    )
    @test Bool(ok_jit) == ok_expected
    @test SciMLBase.successful_retcode(sol_jit.retcode) == ok_expected
end

@testset "compiled DF-SANE on u.^3 .- 2" begin
    u0 = [3.0, -2.0]
    maxiters = 50
    u_host, steps_host, done_host = dfsane(cube, u0, nothing, maxiters)
    @test done_host
    # `solve(NonlinearProblem(cube, u0), DFSane()).stats.nsteps` with NonlinearSolve 4.32
    @test steps_host == 12
    @test u_host ≈ fill(cbrt(2.0), 2)

    u_jit, steps_jit, done_jit = @jit dfsane(cube, Reactant.to_rarray(u0), nothing, maxiters)
    @test Bool(done_jit)
    @test Int(steps_jit) == steps_host
    @test Array(u_jit) ≈ u_host

    # the unit spectral step has not converged within the same number of steps
    _, _, done_unit = dfsane(cube, u0, nothing, steps_host; unit_step = true)
    @test !done_unit
    _, _, done_unit_jit = @jit dfsane(
        cube, Reactant.to_rarray(u0), nothing, steps_host; unit_step = true
    )
    @test !Bool(done_unit_jit)
end
