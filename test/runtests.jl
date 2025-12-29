using ReTestItems, LineSearch, Hwloc, InteractiveUtils
using Pkg

@info sprint(InteractiveUtils.versioninfo)

const GROUP = lowercase(get(ENV, "GROUP", "All"))

function activate_enzyme_env()
    Pkg.activate(joinpath(@__DIR__, "enzyme"))
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

if GROUP == "enzyme"
    # Run Enzyme tests in a separate environment
    # Skip on Julia 1.12+ and pre-release versions due to compatibility issues
    # See https://github.com/SciML/LineSearch.jl/issues/31
    if VERSION >= v"1.12-" || !isempty(VERSION.prerelease)
        @info "Skipping Enzyme tests on Julia $(VERSION) (v1.12+ or pre-release)"
    else
        activate_enzyme_env()
        include(joinpath(@__DIR__, "enzyme", "enzyme_tests.jl"))
    end
else
    const RETESTITEMS_NWORKERS = parse(
        Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Hwloc.num_physical_cores(), 4))))
    const RETESTITEMS_NWORKER_THREADS = parse(Int,
        get(ENV, "RETESTITEMS_NWORKER_THREADS",
            string(max(Hwloc.num_virtual_cores() ÷ RETESTITEMS_NWORKERS, 1))))

    @info "Running tests for group: $(GROUP) with $(RETESTITEMS_NWORKERS) workers"

    ReTestItems.runtests(LineSearch; tags = (GROUP == "all" ? nothing : [Symbol(GROUP)]),
        nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS,
        testitem_timeout = 3600)
end
