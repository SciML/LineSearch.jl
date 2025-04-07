@concrete struct HagerZhang <: AbstractLineSearchAlgorithm
    autodiff
    c1
    c2
    rho
    epsilon
    gamma
    psi3
    maxstep
    initial_alpha
    maxiters::Int
end

function HagerZhang(; autodiff=nothing, c1=0.1, c2=0.9, rho=5.0,
                    epsilon=1e-6, gamma=0.66, psi3=0.1, maxstep=Inf,
                    initial_alpha=true, maxiters=50)
    return HagerZhang(autodiff, c1, c2, rho, epsilon, gamma, psi3, maxstep, initial_alpha, maxiters)
end


@concrete mutable struct HagerZhangCache <: AbstractLineSearchCache
    # Problem and function data
    f
    p
    ϕ        # closure: computes objective φ(α)
    ϕdϕ      # closure: computes (φ(α), φ'(α))
    deriv_op
    u_cache
    fu_cache
    # History of steps in current line search
    alphas::Vector      # step lengths tried
    values::Vector      # φ values at those steps
    slopes::Vector      # φ' (derivative) values at those steps
    # Step control
    alpha               # current step size (floating T)
    initial_alpha       # initial step size (saved for reinit)
    # Stats and back-reference
    stats::Union{SciMLBase.NLStats, Nothing}
    alg::HagerZhang
    maxiters::Int
end


function CommonSolve.init(prob::AbstractNonlinearProblem, alg::HagerZhang,
                          fu, u; stats=nothing, autodiff=nothing, kwargs...)
    # 1. Determine numeric type
    T = promote_type(eltype(fu), eltype(u))
    # 2. Choose autodiff backend and construct derivative operator
    autodiff = autodiff !== nothing ? autodiff : alg.autodiff
    _, _, deriv_op = construct_jvp_or_vjp_operator(prob, fu, u; autodiff)
    # 3. Allocate cache vectors for trial state
    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    # 4. Define closures for φ(α) and φ+dφ(α)
    ϕ   = @closure (f, p, u, du, α, u_cache, fu_cache) -> begin
              @bb @. u_cache = u + α * du
              fu_cache = SciMLBase.evaluate_f!!(f, fu_cache, u_cache, p)
              SciMLBase.add_nf!(stats)             # increment function eval count
              return norm(fu_cache)^2 / 2          # objective value
          end
    ϕdϕ = @closure (f, p, u, du, α, u_cache, fu_cache, deriv_op) -> begin
              @bb @. u_cache = u + α * du
              fu_cache = SciMLBase.evaluate_f!!(f, fu_cache, u_cache, p)
              SciMLBase.add_nf!(stats)
              # Compute directional derivative via AD or analytic Jacobian:
              deriv = deriv_op(du, u_cache, fu_cache, p)
              obj = norm(fu_cache)^2 / 2
              return obj, deriv                   # (φ, φ')
          end
    # 5. Initial step size α (respect maxstep and initial_alpha)
    u_norm = norm(u, Inf)
    α0 = if u_norm == 0 
            one(T)              # if current u is zero-vector, use step = 1
         else 
            alg.initial_alpha isa Bool ? one(T) : T(alg.initial_alpha)
         end
    α0 = min(α0, T(alg.maxstep) / T(max(u_norm, one(T))))
    # 6. Initialize cache and return
    return HagerZhangCache(prob.f, prob.p, ϕ, ϕdϕ, deriv_op,
                           u_cache, fu_cache,
                           Vector{T}(), Vector{T}(), Vector{T}(),
                           α0, α0, stats, alg, alg.maxiters)
end

function CommonSolve.solve!(cache::HagerZhangCache, u, du)
    T = promote_type(eltype(u), eltype(du))
    # φ0 and dφ0 at alpha = 0
    φ0, dφ0 = cache.ϕdϕ(cache.f, cache.p, u, du, zero(T),
                         cache.u_cache, cache.fu_cache, cache.deriv_op)
    if !(isfinite(φ0) && isfinite(dφ0))
        return LineSearchSolution(zero(T), ReturnCode.Failure)   # non-finite baseline
    end
    if dφ0 >= 0
        return LineSearchSolution(zero(T), ReturnCode.ConvergenceFailure)  # not descent
    end
    # Initialize history
    empty!(cache.alphas); empty!(cache.values); empty!(cache.slopes)
    push!(cache.alphas, zero(T)); push!(cache.values, φ0); push!(cache.slopes, dφ0)
    # Initial trial step
    local α = cache.alpha            # initial guess from cache (already of type T)
    α = min(α, T(cache.alg.maxstep)) # enforce maxstep
    if α <= eps(T)
        return LineSearchSolution(zero(T), ReturnCode.Success)   # step too small (no move)
    end
    # Evaluate at initial α
    φ_α, dφ_α = cache.ϕdϕ(cache.f, cache.p, u, du, α,
                           cache.u_cache, cache.fu_cache, cache.deriv_op)
    # Ensure finite by shrinking via psi3
    iter_finite = 0
    while !(isfinite(φ_α) && isfinite(dφ_α)) && iter_finite < cache.alg.maxiters
        α *= T(cache.alg.psi3)   # reduce step
        φ_α, dφ_α = cache.ϕdϕ(cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache, cache.deriv_op)    # reevaluate
        iter_finite += 1
    end
    if !(isfinite(φ_α) && isfinite(dφ_α))
        # Could not find a finite φ even after reducing step
        return LineSearchSolution(zero(T), ReturnCode.Failure)
    end
    push!(cache.alphas, α); push!(cache.values, φ_α); push!(cache.slopes, dφ_α)
    # Now have two points: index1 (0) and index2 (α)
    phi_lim = φ0 + T(cache.alg.epsilon) * abs(φ0)
    ia = 1; ib = 2
    # Bracketing loop
    local cold = zero(T);  local φ_cold = φ0
    local is_bracketed = false
    local iter = 1
    while !is_bracketed && iter < cache.maxiters
        # Current end point is index ib (the last pushed point)
        α = cache.alphas[ib];  φ_α = cache.values[ib];  dφ_α = cache.slopes[ib]
        if dφ_α >= zero(T)
            # Slope non-negative: bracket found between ia and ib
            ib = length(cache.alphas)
            # choose ia as last index <= phi_lim going backwards
            ia = 1
            for i in (ib-1):-1:1
                if cache.values[i] <= phi_lim
                    ia = i
                    break
                end
            end
            is_bracketed = true
        elseif φ_α > phi_lim
            # Function value increased beyond phi_lim, slope still negative: crest scenario
            ib = length(cache.alphas)
            ia = 1
            # Bisect between ia and ib to find a bracket
            (ia, ib) = bisect!(cache, ia, ib, phi_lim)  # This will evaluate mid-points and update history
            is_bracketed = true
        else
            # Still going downhill and φ not increased significantly: expand further
            cold = α;  φ_cold = φ_α   # save last good point
            if nextfloat(cold) >= T(cache.alg.maxstep)
                # Reached maximum step effectively
                return LineSearchSolution(cold, ReturnCode.Success)
            end
            # Propose new α = α * rho
            α = min(cold * T(cache.alg.rho), T(cache.alg.maxstep))
            φ_α, dφ_α = cache.ϕdϕ(cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache, cache.deriv_op)
            # Check finite again, possibly bisect if not finite
            if !(isfinite(φ_α) && isfinite(dφ_α))
                cache.alg.maxstep = α   # shrink maxstep to current α
                # bisect between cold and α until finite or limit
                local α_hi = α;  local α_lo = cold
                local it_f = 1
                while !(isfinite(φ_α) && isfinite(dφ_α)) && it_f < cache.alg.maxiters && α_hi > nextfloat(α_lo)
                    α = (α_lo + α_hi)/2
                    φ_α, dφ_α = cache.ϕdϕ(cache.f, cache.p, u, du, α, cache.u_cache, cache.fu_cache, cache.deriv_op)
                    if isfinite(φ_α) && isfinite(dφ_α)
                        break
                    end
                    α_hi = α     # shrink upper bound
                    it_f += 1
                end
                if !(isfinite(φ_α) && isfinite(dφ_α))
                    # Failed to find finite in bracket
                    return LineSearchSolution(cold, ReturnCode.Failure)
                end
            end
            # Append this new point and continue loop
            push!(cache.alphas, α); push!(cache.values, φ_α); push!(cache.slopes, dφ_α)
            ib = length(cache.alphas)
        end
        iter += 1
    end  # end bracketing loop

    if !is_bracketed
        # Bracketing failed within maxiters
        return LineSearchSolution(cache.alphas[end], ReturnCode.Failure)
    end

    # Now have bracket [ia, ib] with ia < ib
    while iter < cache.maxiters
        # Current bracket interval
        local a = cache.alphas[ia];  local b = cache.alphas[ib]
        if b - a <= eps(b)
            return LineSearchSolution(a, ReturnCode.Success)  # interval too small
        end
        # Interpolation step to find trial between a and b
        local (is_wolfe, i_new, j_new) = secant2!(cache, ia, ib, phi_lim)
        # secant2! will:
        #  - evaluate φ and φ' at new trial alpha within (a,b)
        #  - add new trial to cache.alphas/values/slopes
        #  - return true/false if Wolfe met, and updated bracket indices.
        if is_wolfe
            # Wolfe conditions satisfied at new trial
            local idx = i_new  # index of the accepted point
            return LineSearchSolution(cache.alphas[idx], ReturnCode.Success)
        else
            # Not yet satisfied: update bracket and continue
            ia = i_new;  ib = j_new
        end
        iter += 1
    end

    # If we exit loop by exceeding iterations:
    return LineSearchSolution(cache.alphas[ia], ReturnCode.Failure)
end

function SciMLBase.reinit!(cache::HagerZhangCache; p=cache.p, stats=cache.stats)
    cache.p = p
    cache.stats = stats
    cache.alpha = cache.initial_alpha
    empty!(cache.alphas)
    empty!(cache.values)
    empty!(cache.slopes)
    return cache
end
