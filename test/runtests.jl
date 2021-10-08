# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package ChangesOfVariables" begin
    include("test_with_ladj.jl")
end # testset
