using SciMLTesting, LineSearch, Test

# ExplicitImports only sees an extension module once its trigger package is loaded, so
# load the weakdeps here to bring LineSearchLineSearchesExt into the QA scan.
using LineSearches

# ExplicitImports silently skips an extension that fails to load, so assert the
# extension modules actually exist rather than trusting a green run_qa.
@testset "Extensions loaded" begin
    @test Base.get_extension(LineSearch, :LineSearchLineSearchesExt) !== nothing
end

run_qa(LineSearch)
