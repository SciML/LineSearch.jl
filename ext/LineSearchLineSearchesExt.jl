module LineSearchLineSearchesExt

using LineSearches: LineSearches

using LineSearch: LineSearch

function LineSearch.LineSearchesJL(;
        method = LineSearches.Static(), autodiff = nothing,
        initial_alpha = true
    )
    return LineSearch.LineSearchesJL(method, initial_alpha, autodiff)
end

end
