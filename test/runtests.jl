using ReTestItems, LineSearch, Hwloc, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

const GROUP = get(ENV, "GROUP", "All")

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Hwloc.num_physical_cores(), 4)))
)
const RETESTITEMS_NWORKER_THREADS = parse(
    Int,
    get(
        ENV, "RETESTITEMS_NWORKER_THREADS",
        string(max(Hwloc.num_virtual_cores() ÷ RETESTITEMS_NWORKERS, 1))
    )
)

# Map GROUP to testitem tags
const GROUP_TAGS = Dict(
    "All" => nothing,  # Run all tests
    "Core" => [:core],  # Native line search tests
    "LineSearchesJL" => [:linesearchesjl],  # LineSearches.jl extension tests
    "QA" => [:qa]  # Quality assurance tests (ExplicitImports)
)

if !haskey(GROUP_TAGS, GROUP)
    error("Unknown test group: $(GROUP). Valid groups: $(join(keys(GROUP_TAGS), ", "))")
end

const TAGS = GROUP_TAGS[GROUP]

@info "Running tests for group: $(GROUP) with $(RETESTITEMS_NWORKERS) workers"

ReTestItems.runtests(
    LineSearch; tags = TAGS,
    nworkers = RETESTITEMS_NWORKERS, nworker_threads = RETESTITEMS_NWORKER_THREADS,
    testitem_timeout = 3600
)
