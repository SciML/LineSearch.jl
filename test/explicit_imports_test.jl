@testitem "Explicit Imports" begin
    using ExplicitImports, LineSearch

    @test check_no_implicit_imports(LineSearch) === nothing
    @test check_no_stale_explicit_imports(LineSearch) === nothing
end
