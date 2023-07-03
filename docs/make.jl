# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using ChangesOfVariables

# Doctest setup
DocMeta.setdocmeta!(
    ChangesOfVariables,
    :DocTestSetup,
    :(using ChangesOfVariables, InverseFunctions);
    recursive=true,
)

makedocs(
    sitename = "ChangesOfVariables",
    modules = [ChangesOfVariables],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://JuliaMath.github.io/ChangesOfVariables.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    strict = !("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/JuliaMath/ChangesOfVariables.jl.git",
    forcepush = true,
    push_preview = true,
)
