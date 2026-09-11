using LineSearch, SciMLBase, BenchmarkTools
using DifferentiationInterface, ForwardDiff, ADTypes

const SUITE = BenchmarkGroup()

# Newton-Raphson driver exercising a line search method (mirrors the package's
# own test harness)
function newton_raphson(prob, ls, niter = 20)
    u = copy(prob.u0)
    fu = prob.f(u, prob.p)
    ls_cache = init(prob, ls, fu, u; autodiff = AutoForwardDiff())
    alphas = Float64[]
    for _ in 1:niter
        maximum(abs, fu) < 1.0e-8 && break
        J = DifferentiationInterface.jacobian(
            prob.f, AutoForwardDiff(), u, Constant(prob.p)
        )
        δu = -J \ fu
        ls_sol = solve!(ls_cache, u, δu)
        push!(alphas, ls_sol.step_size)
        @. u = u + ls_sol.step_size * δu
        fu = prob.f(u, prob.p)
    end
    return alphas
end

nlf(x, p) = x .^ 2 .- p
nlp = NonlinearProblem(nlf, [-1.0, 1.0], [3.0])

# =============================================================================
# Line search algorithms inside Newton iteration
# =============================================================================

SUITE["line_search"] = BenchmarkGroup()

SUITE["line_search"]["backtracking"] = @benchmarkable newton_raphson(
    $nlp, BackTracking()
)
SUITE["line_search"]["li_fukushima"] = @benchmarkable newton_raphson(
    $nlp, LiFukushimaLineSearch()
)
SUITE["line_search"]["robust_nonmonotone"] = @benchmarkable newton_raphson(
    $nlp, RobustNonMonotoneLineSearch()
)
SUITE["line_search"]["golden_section"] = @benchmarkable newton_raphson(
    $nlp, GoldenSection(; tol = 1.0e-4)
)
SUITE["line_search"]["hager_zhang"] = @benchmarkable newton_raphson(
    $nlp, HagerZhangLineSearch()
)

# =============================================================================
# Cache init + solve!
# =============================================================================

SUITE["cache"] = BenchmarkGroup()

u0 = [-1.0, 1.0]
fu0 = nlp.f(u0, nlp.p)
ls_cache = init(nlp, BackTracking(), fu0, u0; autodiff = AutoForwardDiff())
δu = [0.5, -0.5]

SUITE["cache"]["init"] = @benchmarkable init(
    $nlp, BackTracking(), $fu0, $u0; autodiff = AutoForwardDiff()
)
SUITE["cache"]["solve!"] = @benchmarkable solve!($ls_cache, $u0, $δu)
