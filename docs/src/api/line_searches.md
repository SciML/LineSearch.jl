# LineSearches.jl

This is an extension for importing line search algorithms from LineSearches.jl into the SciML
interface. Note that these solvers do not come by default, and thus one needs to install the
package before using these solvers:

```julia
using Pkg
Pkg.add("LineSearches")
using LineSearches, LineSearch
```

## Line Search API

!!! tip
    
    Unlike `LineSearches.jl`, we automatically construct the gradient/jacobian functionality
    from the problem specification using automatic differentiation (if analytic versions
    are not provided).

```@docs
LineSearchesJL
```
