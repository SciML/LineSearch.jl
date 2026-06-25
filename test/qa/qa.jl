using SciMLTesting, LineSearch, Test

run_qa(
    LineSearch;
    explicit_imports = true,
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                :Failure, :Success, :T,         # SciMLBase.ReturnCode (not public)
                :NLStats, :has_jac, :has_jvp, :has_vjp,  # SciMLBase (not public)
                :ForwardMode, :mode,            # ADTypes (not public)
                :init, :solve!,                 # CommonSolve (not public)
                :get_extension,                 # Base (not public)
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractNonlinearProblem,      # SciMLBase (not public)
            ),
        ),
    )
)
