"""
    GoldenSection()

A derivative-free line search method that minimizes a unimodal function by 
successively narrowing the range of values inside which the local minimum must lie.
"""
struct GoldenSection{T<:AbstractFloat}
    tol::T
    maxiter::Int
end

# Default constructor
GoldenSection(; tol=1e-7, maxiter=100) = GoldenSection(tol, maxiter)

function (gs::GoldenSection)(f, x, d, α_initial, f_x, g_x)
    T = typeof(α_initial)
    
    # Defines the 1D objective: ϕ(α) = f(x + αd)
    ϕ(α) = f(x + α .* d)

    a = T(0.0)
    b = α_initial
    
    # Golden ratio constants
    φ = (sqrt(T(5)) + 1) / 2
    resphi = 2 - φ  # ~0.382

    # Initial internal points
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    
    f1 = ϕ(x1)
    f2 = ϕ(x2)

    α_best = α_initial
    f_best = f_x
    iter = 0

    while abs(b - a) > gs.tol && iter < gs.maxiter
        iter += 1
        if f1 < f2
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = ϕ(x1)
        else
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = ϕ(x2)
        end
    end

    α_best = (a + b) / 2
    f_best = ϕ(α_best)
    
    return α_best, f_best
end