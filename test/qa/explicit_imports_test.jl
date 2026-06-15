using ExplicitImports, LineSearch, Test

@testset "Explicit Imports" begin
    @test check_no_implicit_imports(LineSearch) === nothing
    @test check_no_stale_explicit_imports(LineSearch) === nothing
end
