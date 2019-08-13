using Documenter, DiscriminantAnalysis

makedocs(
    format = :html,
    sitename = "DiscriminantAnalysis.jl",
    authors = "Tim Thatcher",
    pages = Any[
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Interface" => "interface.md"
    ]
)

deploydocs(
    repo = "github.com/trthatcher/DiscriminantAnalysis.jl.git"
)