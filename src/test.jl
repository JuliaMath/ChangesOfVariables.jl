# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).


_generalized_logabsdet(A) = logabsdet(A)
_generalized_logabsdet(x::Real) = log(abs(x))

function _auto_with_logabsdet_jacobian(f, x, getjacobian, rv_and_back)
    y = f(x)
    V, to_x = rv_and_back(x)
    vf(V) = rv_and_back(f(to_x(V)))[1]
    ladj = _generalized_logabsdet(getjacobian(vf, V))[1]
    return (y, ladj)
end


"""
    ChangesOfVariables.test_with_logabsdet_jacobian(
        f, x, getjacobian, rv_and_back = x -> (x, identity);
        compare = isapprox, test_inferred::Bool = true, kwargs...
    )

Test if [`with_logabsdet_jacobian(f, x)`](@ref) is implemented correctly.

Checks if the result of `with_logabsdet_jacobian(f, x)` is approximately
equal to `(f(x), logabsdet(getjacobian(f, x)))`
        
So the test uses `getjacobian(f, x)` to calculate a reference Jacobian for
`f` at `x`. Passing `ForwardDiff.jabobian`, `Zygote.jacobian` or similar as
the `getjacobian` function will do fine in most cases. If input and output
of `f` are real scalar values, use `ForwardDiff.derivative`.

If `getjacobian(f, x)` can't handle the type of `x` of `f(x)` because they
are not real-valued vectors, use the `rv_and_back` argument to pass a
function with the following behavior

```julia
v, back = rv_and_back(x)
v isa AbstractVector{<:Real}
back(v) == x
```

If `test_inferred == true`, type inference on `with_logabsdet_jacobian` will
be tested.

`kwargs...` are forwarded to `compare`.
"""
function test_with_logabsdet_jacobian(
    f, x, getjacobian, rv_and_back = x -> (x, identity);
    compare = isapprox, test_inferred::Bool = true, kwargs...
)
    @testset "test_with_logabsdet_jacobian: $f with input $x" begin
        y, ladj = if test_inferred
            @inferred with_logabsdet_jacobian(f, x)
        else
            with_logabsdet_jacobian(f, x)
        end
        ref_y, ref_ladj = _auto_with_logabsdet_jacobian(f, x, getjacobian, rv_and_back)
        @test compare(y, ref_y; kwargs...)
        @test compare(ladj, ref_ladj; kwargs...)
    end
    return nothing
end
