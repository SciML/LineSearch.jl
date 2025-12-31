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

# Function barrier for JVP operator to avoid type instability from captured variables
@inline function _make_jvp_deriv_op(jvp_op)
    return @closure (du, u, fu, p) -> dot(fu, jvp_op(du, u, p))
end

# Function barrier for VJP operator to avoid type instability from captured variables
@inline function _make_vjp_deriv_op(vjp_op)
    return @closure (du, u, fu, p) -> dot(du, vjp_op(fu, u, p))
end

function construct_jvp_or_vjp_operator(prob::AbstractNonlinearProblem, fu, u; autodiff)
    if SciMLBase.has_jvp(prob.f)
        jvp_op = JacVecOperator(prob, fu, u; autodiff)
        return jvp_op, nothing, _make_jvp_deriv_op(jvp_op)
    elseif SciMLBase.has_vjp(prob.f)
        vjp_op = VecJacOperator(prob, fu, u; autodiff)
        return nothing, vjp_op, _make_vjp_deriv_op(vjp_op)
    elseif u isa Number && SciMLBase.has_jac(prob.f)
        jvp_op = JacVecOperator(prob, fu, u; autodiff)
        return jvp_op, nothing, _make_jvp_deriv_op(jvp_op)
    elseif autodiff isa ADTypes.AbstractADType
        if ADTypes.mode(autodiff) isa ADTypes.ForwardMode
            jvp_op = JacVecOperator(prob, fu, u; autodiff)
            return jvp_op, nothing, _make_jvp_deriv_op(jvp_op)
        else
            vjp_op = VecJacOperator(prob, fu, u; autodiff)
            return nothing, vjp_op, _make_vjp_deriv_op(vjp_op)
        end
    elseif SciMLBase.has_jac(prob.f)
        jvp_op = JacVecOperator(prob, fu, u; autodiff)
        return jvp_op, nothing, _make_jvp_deriv_op(jvp_op)
    else
        error("Exhausted all possibilities for autodiff and analytic jacobian/jvp/vjp \
               options. Either specify `autodiff` while constructing `LineSearchesJL` or \
               pass it to `init` as a keyword argument.")
    end
end
