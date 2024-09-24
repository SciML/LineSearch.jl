evaluate_f!!(prob::AbstractNonlinearProblem, fu, u, p) = evaluate_f!!(prob.f, fu, u, p)

function evaluate_f!!(f::NonlinearFunction, fu, u, p)
    if SciMLBase.isinplace(f)
        f(fu, u, p)
        return fu
    end
    return f(u, p)
end

add_nf!(::Nothing, _ = 1) = nothing
add_nf!(stats::SciMLBase.NLStats, nf = 1) = stats.nf += nf
