# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).

import Test
import ChangesOfVariables
import Documenter

Test.@testset "Package ChangesOfVariables" begin
    include("test_test.jl")
    include("test_with_ladj.jl")
    include("test_setladj.jl")

    # doctests
    Documenter.DocMeta.setdocmeta!(
        ChangesOfVariables,
        :DocTestSetup,
        :(using ChangesOfVariables, InverseFunctions);
        recursive=true,
    )
    Documenter.doctest(ChangesOfVariables)
end # testset
