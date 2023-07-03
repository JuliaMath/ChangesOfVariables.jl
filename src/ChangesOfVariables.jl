# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).

"""
    ChangesOfVariables

Lightweight package that defines functionality to calculate volume element
changes for functions that perform a change of variables (like coordinate
transformations).
"""
module ChangesOfVariables

using LinearAlgebra
using Test

include("with_ladj.jl")
include("setladj.jl")
include("test.jl")

@static if !isdefined(Base, :get_extension)
    include("../ext/ChangesOfVariablesInverseFunctionsExt.jl")
end

end # module
