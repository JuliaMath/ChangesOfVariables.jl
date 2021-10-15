# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).


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
"""
function test_with_logabsdet_jacobian(f, x, getjacobian; compare = isapprox, kwargs...)
    @testset "test_with_logabsdet_jacobian: $f with input $x" begin
        ref_y, test_type_inference = try
            @inferred(f(x)), true
        catch err
            f(x), false
        end

        y, ladj = if test_type_inference
            @inferred with_logabsdet_jacobian(f, x)
        else
            with_logabsdet_jacobian(f, x)
        end

        ref_ladj = _generalized_logabsdet(getjacobian(f, x))[1]
    
        @test compare(y, ref_y; kwargs...)
        @test compare(ladj, ref_ladj; kwargs...)
    end
    return nothing
end


_generalized_logabsdet(A) = logabsdet(A)
_generalized_logabsdet(x::Real) = log(abs(x))
