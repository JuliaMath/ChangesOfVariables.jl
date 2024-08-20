# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).

"""
    ChangesOfVariables

Lightweight package that defines functionality to calculate volume element
changes for functions that perform a change of variables (like coordinate
transformations).
"""
module ChangesOfVariables

using LinearAlgebra

include("with_ladj.jl")
include("setladj.jl")

"""
    ChangesOfVariables.test_with_logabsdet_jacobian(f, x, getjacobian; compare = isapprox, kwargs...)

Test if [`with_logabsdet_jacobian(f, x)`](@ref) is implemented correctly.

Checks if the result of `with_logabsdet_jacobian(f, x)` is approximately
equal to `(f(x), logabsdet(getjacobian(f, x)))`
        
So the test uses `getjacobian(f, x)` to calculate a reference Jacobian for
`f` at `x`. Passing `ForwardDiff.jabobian`, `Zygote.jacobian` or similar as
the `getjacobian` function will do fine in most cases. If input and output
of `f` are real scalar values, use `ForwardDiff.derivative`.

Note that the result of `getjacobian(f, x)` must be a real-valued matrix
or a real scalar, so you may need to use a custom `getjacobian` function
that transforms the shape of `x` and `f(x)` internally, in conjunction
with automatic differentiation.

`kwargs...` are forwarded to `compare`.

!!! Note
    On Julia >= 1.9, you have to load the `Test` standard library to be able to use
    this function.
"""
function test_with_logabsdet_jacobian end

@static if !isdefined(Base, :get_extension)
    include("../ext/ChangesOfVariablesInverseFunctionsExt.jl")
    include("../ext/ChangesOfVariablesTestExt.jl")
end

# Better error message if users forget to load Test
if isdefined(Base, :get_extension) && isdefined(Base.Experimental, :register_error_hint)
    function __init__()
        Base.Experimental.register_error_hint(MethodError) do io, exc, _, _
            if exc.f === test_with_logabsdet_jacobian &&
                (Base.get_extension(ChangesOfVariables, :ChangesOfVariablesTest) === nothing)
                print(io, "\nDid you forget to load Test?")
            end
        end
    end
end

end # module
