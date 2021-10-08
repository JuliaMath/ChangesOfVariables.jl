# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).

using ChangesOfVariables
using Test

using LinearAlgebra
using ForwardDiff: derivative, jacobian


fwddiff_ladj(f, x::Real) = log(abs(derivative(f, x)))
fwddiff_ladj(f, x::AbstractArray{<:Real}) = logabsdet(jacobian(f, x))[1]
fwddiff_with_ladj(f, x) = (f(x), fwddiff_ladj(f, x))

ascomplex(A::AbstractArray{T}) where T = reinterpret(Complex{T}, A)
asreal(A::AbstractArray{Complex{T}}) where T = reinterpret(T, A)

isaprx(a, b) = isapprox(a,b)
isaprx(a::NTuple{N,Any}, b::NTuple{N,Any}) where N = all(map(isaprx, a, b))


foo(x) = inv(exp(-x) + 1)

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(foo), x)
    y = foo(x)
    ladj = -x + 2 * log(y)
    (y, ladj)
end


@testset "with_logabsdet_jacobian" begin
    x = 4.2
    X = rand(10)
    A = rand(5, 5)
    CA = rand(10, 5)

    @test isaprx(@inferred(with_logabsdet_jacobian(foo, x)), fwddiff_with_ladj(foo, x))

    @static if VERSION >= v"1.6"
        log_foo = log ∘ foo
        @test isaprx(@inferred(with_logabsdet_jacobian(log_foo, x)), fwddiff_with_ladj(log_foo, x))
    end

    mapped_foo = Base.Fix1(map, foo)
    @test isaprx(@inferred(with_logabsdet_jacobian(mapped_foo, x)), fwddiff_with_ladj(mapped_foo, x))
    @test isaprx(@inferred(with_logabsdet_jacobian(mapped_foo, fill(x))), fwddiff_with_ladj(mapped_foo, fill(x)))
    @test isaprx(@inferred(with_logabsdet_jacobian(mapped_foo, Ref(x))), fwddiff_with_ladj(mapped_foo, fill(x)))
    @test isaprx(@inferred(with_logabsdet_jacobian(mapped_foo, (x,))), (mapped_foo((x,)), fwddiff_ladj(mapped_foo, x)))
    @test isaprx(@inferred(with_logabsdet_jacobian(mapped_foo, X)), fwddiff_with_ladj(mapped_foo, X))
    
    broadcasted_foo = Base.Fix1(broadcast, foo)
    @test isaprx(@inferred(with_logabsdet_jacobian(broadcasted_foo, x)), fwddiff_with_ladj(broadcasted_foo, x))
    @test isaprx(@inferred(with_logabsdet_jacobian(broadcasted_foo, fill(x))), fwddiff_with_ladj(broadcasted_foo, x))
    @test isaprx(@inferred(with_logabsdet_jacobian(broadcasted_foo, Ref(x))), fwddiff_with_ladj(broadcasted_foo, x))
    @test isaprx(@inferred(with_logabsdet_jacobian(broadcasted_foo, (x,))), (mapped_foo((x,)), fwddiff_ladj(mapped_foo, x)))
    @test isaprx(@inferred(with_logabsdet_jacobian(broadcasted_foo, X)), fwddiff_with_ladj(broadcasted_foo, X))

    for f in (identity, adjoint, transpose)
        @test isaprx(@inferred(with_logabsdet_jacobian(f, x)), fwddiff_with_ladj(f, x))
        @test isaprx(@inferred(with_logabsdet_jacobian(f, A)), fwddiff_with_ladj(f, A))
    end
    
    @test isaprx(@inferred(with_logabsdet_jacobian(inv, x)), fwddiff_with_ladj(inv, x))
    @test isaprx(@inferred(with_logabsdet_jacobian(inv, A)), fwddiff_with_ladj(inv, A))
    @test isaprx(@inferred(with_logabsdet_jacobian(inv, ascomplex(CA))), (inv(ascomplex(CA)), fwddiff_ladj(CA -> asreal(inv(ascomplex(CA))), CA)))

    for f in (exp, log, exp2, log2, exp10, log10, expm1, log1p)
        @test isaprx(@inferred(with_logabsdet_jacobian(f, x)), fwddiff_with_ladj(f, x))
    end
end
