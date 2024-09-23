@concrete struct LineSearchesJL <: AbstractLineSearchAlgorithm
    method
    initial_alpha
    autodiff <: Union{Nothing, ADTypes.AbstractADType}
end

function CommonSolve.init(prob::AbstractNonlinearProblem, alg::LineSearchesJL; kwargs...)
    T = eltype(prob.u0)
    f = prob.f
    if prob.u isa Number
    else
        # Both forward and reverse AD can be used for line-search.
        # We prefer forward AD for better performance, however, reverse AD is also
        # supported if user explicitly requests it.
        # 1. If jvp is available, we use forward AD;
        # 2. If vjp is available, we use reverse AD;
        # 3. If reverse type is requested, we use reverse AD;
        # 4. Finally, we use forward AD.
        if SciMLBase.has_jvp(prob)
        elseif SciMLBase.has_vjp(prob)
        elseif alg.autodiff isa ADTypes.AbstractADType
        else
            forward_ad = get_forward_mode_ad(prob)
            reverse_ad = get_reverse_mode_ad(prob)
            if forward_ad === nothing && reverse_ad === nothing
                error("No suitable automatic differentiation backend found. Backends \
                       checked: $(join(FORWARD_AD_ORDERING, ", ")) & \
                       $(join(REVERSE_AD_ORDERING, ", "))")
            end
            ad = if (forward_ad isa ADTypes.AutoFiniteDiff || forward_ad === nothing) &&
                    reverse_ad !== nothing
                reverse_ad
            else
                forward_ad
            end
            if ADTypes.mode(ad) isa ADTypes.ForwardMode # Using `pushforward`
                if SciMLBase.isinplace(prob)
                    extras = DI.prepare_pushforward
                else
                end
            else # Using `pullback`
            end
        end
    end
end
