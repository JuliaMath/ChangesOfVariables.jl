# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).


_to_realvec_and_back(V::AbstractVector{<:Real}) = V, identity
_to_realvec_and_back(x::Real) = [x], V -> V[1]
_to_realvec_and_back(x::Complex) = [real(x), imag(x)], V -> Complex(V[1], V[2])
_to_realvec_and_back(x::NTuple{N}) where N = [x...], V -> ntuple(i -> V[i], Val(N))

function _to_realvec_and_back(x::Ref)
    xval = x[]
    V, to_xval = _to_realvec_and_back(xval)
    back_to_ref(V) = Ref(to_xval(V))
    return (V, back_to_ref)
end

_to_realvec_and_back(A::AbstractArray{<:Real}) = vec(A), V -> reshape(V, size(A))

function _to_realvec_and_back(A::AbstractArray{Complex{T}, N}) where {T<:Real, N}
    RA = cat(real.(A), imag.(A), dims = N+1) 
    V, to_array = _to_realvec_and_back(RA)
    function back_to_complex(V)
        RA = to_array(V)
        Complex.(view(RA, map(_ -> :, size(A))..., 1), view(RA, map(_ -> :, size(A))..., 2))
    end
    return (V, back_to_complex)
end


_to_realvec(x) = _to_realvec_and_back(x)[1]


function _auto_with_logabsdet_jacobian(f, x, getjacobian)
    y = f(x)
    V, to_x = _to_realvec_and_back(x)
    vf(V) = _to_realvec(f(to_x(V)))
    ladj = logabsdet(getjacobian(vf, V))[1]
    return (y, ladj)
end


"""
    ChangesOfVariables.test_with_logabsdet_jacobian(
        f, x, getjacobian;
        test_inferred::Bool = true, kwargs...
    )

Test if [`with_logabsdet_jacobian(f, x)`](@ref) is implemented correctly.

Checks if the result of `with_logabsdet_jacobian(f, x)` is approximately
equal to `(f(x), logabsdet(getjacobian(f, x)))`
        
So the test uses `getjacobian(f, x)` to calculate a reference Jacobian for
`f` at `x`. Passing `ForwardDiff.jabobian`, `Zygote.jacobian` or similar as
the `getjacobian` function will do fine in most cases.

If `x` or `f(x)` are real-valued scalars or complex-valued scalars or arrays,
the test will try to reshape them automatically, to account for limitations
of (e.g.) `ForwardDiff` and to ensure the result of `getjacobian` is a real
matrix.

If `test_inferred == true` will test type inference on
`with_logabsdet_jacobian`.

`kwargs...` are forwarded to `isapprox`.
"""
function test_with_logabsdet_jacobian(f, x, getjacobian; compare=isapprox, test_inferred::Bool = true, kwargs...)
    @testset "test_with_logabsdet_jacobian: $f with input $x" begin
        y, ladj = if test_inferred
            @inferred with_logabsdet_jacobian(f, x)
        else
            with_logabsdet_jacobian(f, x)
        end
        ref_y, ref_ladj = _auto_with_logabsdet_jacobian(f, x, getjacobian)
        @test compare(y, ref_y; kwargs...)
        @test compare(ladj, ref_ladj; kwargs...)
    end
    return nothing
end
