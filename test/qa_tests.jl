@testitem "Aqua" tags=[:nopre] begin
    using Aqua, LineSearch

    Aqua.test_all(LineSearch; ambiguities = false, piracies = false)
    Aqua.test_ambiguities(LineSearch; recursive = false)
end
