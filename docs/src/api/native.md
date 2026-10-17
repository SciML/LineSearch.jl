# Native Line Search Algorithms

## Result Type

```@docs
LineSearchSolution
```

## Merit Functions

A line search algorithm consumes only `ϕ(α)` and `ϕ'(α)` along the ray
`u + α ⋅ du`; the merit is what those mean for a given caller. Separating them
lets one implementation serve both root finding and optimization.

```@docs
AbstractMerit
ResidualMerit
ObjectiveMerit
```

## Controlling the Search

```@docs
set_initial_step!
```

## No Line Search

```@docs
NoLineSearch
```

## Derivative-Free Line Searches

```@docs
GoldenSection
LiFukushimaLineSearch
RobustNonMonotoneLineSearch
```

## Backtracking Line Search

```@docs
BackTracking
ArmijoLineSearch
ProjectedBackTracking
get_trial
StrongWolfeLineSearch
```

## Wolfe Line Searches

```@docs
HagerZhangLineSearch
MoreThuenteLineSearch
```

`ArmijoLineSearch` reduces the step by a fixed factor, whereas `BackTracking` uses
quadratic or cubic interpolation. Both enforce sufficient decrease; neither
enforces a Wolfe curvature condition. `ProjectedBackTracking` uses the actual
box-projected displacement in that condition and can cross multiple bound hits.
Its cached trial point must be used by the caller, since the accepted path is
piecewise linear rather than a ray.
