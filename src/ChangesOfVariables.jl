# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).

"""
    ChangesOfVariables

Lightweight package that defines functionality to calculate volume element
changes for functions that perform a change of variables (like coordinate
transformations).
"""
module ChangesOfVariables

using ChainRulesCore
using LinearAlgebra
using Test

include("with_ladj.jl")
include("test.jl")

end # module
