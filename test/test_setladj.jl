# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).

using Test
using ChangesOfVariables
using InverseFunctions

const ChangesOfVariablesInverseFunctionsExt = if isdefined(Base, :get_extension)
    Base.get_extension(ChangesOfVariables, :ChangesOfVariablesInverseFunctionsExt)
else
    ChangesOfVariables.ChangesOfVariablesInverseFunctionsExt
end
const InverseFunctionWithLADJ = ChangesOfVariablesInverseFunctionsExt.InverseFunctionWithLADJ

include("getjacobian.jl")


# Dummy testing type that looks like something that represents abstract zeros:
struct _Zero{T} end
_Zero(::T) where {T} = _Zero{T}()


@testset "setladj" begin
    @test @inferred(setladj(Real, _Zero)) isa ChangesOfVariables.FunctionWithLADJ{Type{Real},Type{_Zero}}
    @test @inferred(ChangesOfVariables.FunctionWithLADJ(Real, _Zero)) isa ChangesOfVariables.FunctionWithLADJ{Type{Real},Type{_Zero}}
    @test @inferred(ChangesOfVariables.FunctionWithLADJ(widen, _Zero)) isa ChangesOfVariables.FunctionWithLADJ{typeof(widen),Type{_Zero}}
    @test @inferred(ChangesOfVariables.FunctionWithLADJ(Real, zero)) isa ChangesOfVariables.FunctionWithLADJ{Type{Real},typeof(zero)}
    @test @inferred(ChangesOfVariables.FunctionWithLADJ(widen, zero)) isa ChangesOfVariables.FunctionWithLADJ{typeof(widen),typeof(zero)}

    @test @inferred(InverseFunctionWithLADJ(Real, _Zero)) isa InverseFunctionWithLADJ{Type{Real},Type{_Zero}}
    @test @inferred(InverseFunctionWithLADJ(widen, _Zero)) isa InverseFunctionWithLADJ{typeof(widen),Type{_Zero}}
    @test @inferred(InverseFunctionWithLADJ(Real, zero)) isa InverseFunctionWithLADJ{Type{Real},typeof(zero)}
    @test @inferred(InverseFunctionWithLADJ(widen, zero)) isa InverseFunctionWithLADJ{typeof(widen),typeof(zero)}

    @test @inferred(setladj(setladj(exp, x -> 0), x -> x)) isa ChangesOfVariables.FunctionWithLADJ{typeof(exp)}
    ChangesOfVariables.test_with_logabsdet_jacobian(setladj(setladj(exp, x -> 0), x -> x), 1.7, getjacobian)

    x = 4.2
    y = x^2

    f_fwd = setladj(x -> x^2, x -> log(2*x))
    f_inv = setladj(y -> sqrt(y), y -> log(inv(2*sqrt(y))))
    ChangesOfVariables.test_with_logabsdet_jacobian(f_fwd, x, getjacobian)
    ChangesOfVariables.test_with_logabsdet_jacobian(f_inv, y, getjacobian)

    f = @inferred setladj(setinverse(x -> x^2, x -> sqrt(x)), x -> log(2*x))
    @test @inferred(f(x)) == y
    ChangesOfVariables.test_with_logabsdet_jacobian(f, x, getjacobian)
    ChangesOfVariables.test_with_logabsdet_jacobian(inverse(f), y, getjacobian)
    ChangesOfVariables.test_with_logabsdet_jacobian(inverse(inverse(f)), x, getjacobian)
    @inferred(inverse(inverse(f))) isa ChangesOfVariables.FunctionWithLADJ

    @static if isdefined(InverseFunctions, :setinverse)
        g = setinverse(f_fwd, f_inv)
        ChangesOfVariables.test_with_logabsdet_jacobian(g, x, getjacobian)
        ChangesOfVariables.test_with_logabsdet_jacobian(inverse(g), y, getjacobian)
        ChangesOfVariables.test_with_logabsdet_jacobian(inverse(inverse(g)), x, getjacobian)
    end
end
