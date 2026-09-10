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

    deriv_op = _get_deriv_op(jvp_op, vjp_op)

    return jvp_op, vjp_op, deriv_op
end

# `jv` is a preallocated buffer (`similar(fu)` for JVP, `similar(u)` for VJP), or
# `nothing` for scalar problems where the operator has no in-place form. Using the
# in-place JacobianOperator API avoids the `zero(output_cache)` allocation that the
# out-of-place call performs on every evaluation.
function _get_deriv_op(jvp_op, vjp_op)
    return @closure (jv, du, u, fu, p) -> begin
        # Immutable buffers (SArray) and scalars have no in-place JVP form; the
        # out-of-place call is heap-free for those types.
        if jv === nothing || jv isa Union{Number, SArray}
            return dot(fu, jvp_op(du, u, p))
        end
        jvp_op(jv, du, u, p)
        return dot(fu, jv)
    end
end

function _get_deriv_op(jvp_op::Nothing, vjp_op)
    return @closure (jv, du, u, fu, p) -> begin
        if jv === nothing || jv isa Union{Number, SArray}
            return dot(du, vjp_op(fu, u, p))
        end
        vjp_op(jv, fu, u, p)
        return dot(du, jv)
    end
end

function residual_jv_cache(jvp_op, vjp_op, fu, u)
    # Numbers and immutable static arrays cannot be written into by the in-place
    # JacobianOperator API; `_get_deriv_op` falls back to out-of-place for them.
    (u isa Number || fu isa Number || fu isa SArray || u isa SArray) && return nothing
    if jvp_op !== nothing
        @bb jv = similar(fu)
        return jv
    end
    @bb jv = similar(u)
    return jv
end
