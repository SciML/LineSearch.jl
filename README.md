# LineSearch.jl: High-Performance Unified Line Search Algorithms

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/LineSearch/stable/)

[![codecov](https://codecov.io/gh/SciML/LineSearch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SciML/LineSearch.jl)
[![Build Status](https://github.com/SciML/LineSearch.jl/workflows/CI/badge.svg)](https://github.com/SciML/LineSearch.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

LineSearch.jl is a unified interface for the line search packages of Julia. The package
includes its own high-performance line search algorithms.

Performance is key: the current methods are made to be highly performant on scalar and
statically sized small problems, with options for large-scale systems. If you run into any
performance issues, please file an issue.

> [!WARNING]
> Currently this package is meant to be more developer focused. Most users are recommended
> to use this functionality via NonlinearSolve.jl. Support for other packages in the
> ecosystem like Optimization.jl is planned for the future.

## Installation

To install LineSearch.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("LineSearch")
```
