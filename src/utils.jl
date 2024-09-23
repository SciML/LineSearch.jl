const FORWARD_AD_ORDERING = [
    ADTypes.AutoPolyesterForwardDiff(),
    ADTypes.AutoForwardDiff(),
    ADTypes.AutoEnzyme(; mode = EnzymeCore.Forward),
    ADTypes.AutoFiniteDiff()
]

const REVERSE_AD_ORDERING = [
    ADTypes.AutoEnzyme(; mode = EnzymeCore.Reverse),
    ADTypes.AutoZygote(),
    ADTypes.AutoReverseDiff(),
    ADTypes.AutoTracker(),
    ADTypes.AutoFiniteDiff()
]

function get_forward_mode_ad(_, ad::ADTypes.AbstractADType; force::Bool = false)
    @assert check_forward_mode(ad) "`ad` must be a forward mode AD"
    return ad
end
get_forward_mode_ad(prob, ::Nothing; force::Bool = false) = get_forward_mode_ad(prob; force)
function get_forward_mode_ad(prob; force::Bool = false)
    ad = if SciMLBase.isinplace(prob)
        findfirst(FORWARD_AD_ORDERING) do backend
            DI.check_available(backend) && DI.check_twoarg(backend)
        end
    else
        findfirst(DI.check_available, FORWARD_AD_ORDERING)
    end
    if force
        @assert ad!==nothing "No suitable forward mode automatic differentiation backend \
                              found. Backends checked: $(join(FORWARD_AD_ORDERING, ", "))"
    end
    return ad
end

function get_reverse_mode_ad(_, ad::ADTypes.AbstractADType; force::Bool = false)
    @assert check_reverse_mode(ad) "`ad` must be a reverse mode AD"
    return ad
end
get_reverse_mode_ad(prob, ::Nothing; force::Bool = false) = get_reverse_mode_ad(prob; force)
function get_reverse_mode_ad(prob; force::Bool = false)
    ad = if SciMLBase.isinplace(prob)
        findfirst(REVERSE_AD_ORDERING) do backend
            DI.check_available(backend) && DI.check_twoarg(backend)
        end
    else
        findfirst(DI.check_available, REVERSE_AD_ORDERING)
    end
    if force
        @assert ad!==nothing "No suitable reverse mode automatic differentiation backend \
                              found. Backends checked: $(join(REVERSE_AD_ORDERING, ", "))"
    end
    return ad
end

check_forward_mode(ad::ADTypes.AbstractADType) = ADTypes.mode(ad) isa ADTypes.ForwardMode

check_reverse_mode(ad::AutoFiniteDiff) = true
check_reverse_mode(ad::ADTypes.AbstractADType) = ADTypes.mode(ad) isa ADTypes.ReverseMode
