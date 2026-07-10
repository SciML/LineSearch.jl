using LineSearch
using Test

@testset "public API documentation" begin
    public_names = filter(!=(:LineSearch), names(LineSearch; all = false, imported = false))
    expected_names = Set(
        [
            :BackTracking,
            :GoldenSection,
            :LiFukushimaLineSearch,
            :LineSearchSolution,
            :LineSearchesJL,
            :NoLineSearch,
            :RobustNonMonotoneLineSearch,
            :StrongWolfeLineSearch,
        ]
    )
    @test Set(public_names) == expected_names

    for name in public_names
        binding = Docs.Binding(LineSearch, name)
        @test Docs.hasdoc(binding)
    end

    docs_text = read(joinpath(pkgdir(LineSearch), "docs", "src", "api", "native.md"), String) *
        read(joinpath(pkgdir(LineSearch), "docs", "src", "api", "line_searches.md"), String)
    for name in public_names
        @test occursin(string(name), docs_text)
    end
end
