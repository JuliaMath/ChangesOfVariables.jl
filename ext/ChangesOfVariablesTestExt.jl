module ChangesOfVariablesTestExt

using Test: @inferred, @test, @testset
using ChangesOfVariables: ChangesOfVariables, logabsdet, with_logabsdet_jacobian

function ChangesOfVariables.test_with_logabsdet_jacobian(f, x, getjacobian; compare = isapprox, kwargs...)
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

end # module
