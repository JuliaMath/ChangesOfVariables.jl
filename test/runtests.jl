# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).

using ChangesOfVariables
using Documenter
using Test

Test.@testset "Package ChangesOfVariables" begin
    include("test_with_ladj.jl")

    # doctests
    DocMeta.setdocmeta!(
        ChangesOfVariables,
        :DocTestSetup,
        :(using ChangesOfVariables);
        recursive=true,
    )
    doctest(ChangesOfVariables)
end # testset
