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

function construct_jvp_or_vjp_operator(prob::AbstractNonlinearProblem, fu, u; autodiff)
    if SciMLBase.has_jvp(prob.f)
        jvp_op = JacVecOperator(prob, fu, u; autodiff)
        vjp_op = nothing
    elseif SciMLBase.has_vjp(prob.f)
        vjp_op = VecJacOperator(prob, fu, u; autodiff)
        jvp_op = nothing
    elseif u isa Number && SciMLBase.has_jac(prob.f)
        jvp_op = JacVecOperator(prob, fu, u; autodiff)
        vjp_op = nothing
    elseif autodiff isa ADTypes.AbstractADType
        if ADTypes.mode(autodiff) isa ADTypes.ForwardMode
            jvp_op = JacVecOperator(prob, fu, u; autodiff)
            vjp_op = nothing
        else
            vjp_op = VecJacOperator(prob, fu, u; autodiff)
            jvp_op = nothing
        end
    elseif SciMLBase.has_jac(prob.f)
        jvp_op = JacVecOperator(prob, fu, u; autodiff)
        vjp_op = nothing
    else
        error("Exhausted all possibilities for autodiff and analytic jacobian/jvp/vjp \
               options. Either specify `autodiff` while constructing `LineSearchesJL` or \
               pass it to `init` as a keyword argument.")
    end

    deriv_op = if jvp_op !== nothing
        @closure (du, u, fu, p) -> dot(fu, jvp_op(du, u, p))
    else
        @closure (du, u, fu, p) -> dot(du, vjp_op(fu, u, p))
    end

    return jvp_op, vjp_op, deriv_op
end
