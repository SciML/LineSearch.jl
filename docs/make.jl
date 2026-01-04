using Documenter, DocumenterCitations
using LineSearch

cp(
    joinpath(@__DIR__, "Manifest.toml"),
    joinpath(@__DIR__, "src/assets/Manifest.toml"), force = true
)
cp(
    joinpath(@__DIR__, "Project.toml"),
    joinpath(@__DIR__, "src/assets/Project.toml"), force = true
)

include("pages.jl")

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(;
    sitename = "LineSearch.jl",
    authors = "Avik Pal",
    modules = [LineSearch],
    clean = true,
    doctest = false,
    linkcheck = true,
    checkdocs = :exports,
    warnonly = [:missing_docs],
    plugins = [bib],
    format = Documenter.HTML(
        assets = ["assets/favicon.ico", "assets/citations.css"],
        canonical = "https://docs.sciml.ai/LineSearch/stable/"
    ),
    pages
)

deploydocs(repo = "github.com/SciML/LineSearch.jl.git"; push_preview = true)
