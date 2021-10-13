# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).

using ChangesOfVariables
using Test

using LinearAlgebra
import ForwardDiff

using ChangesOfVariables: test_with_logabsdet_jacobian


include("rv_and_back.jl")

@testset "rv_and_back" begin
    for x in (rand(3), 0.5, Complex(0.2,0.7), (3,5,9), Ref(42), rand(3, 4, 5), Complex.(rand(3,5), rand(3,5)))
        V, to_x = rv_and_back(x)
        @test V isa AbstractVector{<:Real}
        @test V == rv_and_back(x)[1]
        @test x isa Ref ? to_x(V)[] == x[] : to_x(V) == x
    end
end


@testset "test_with_logabsdet_jacobian" begin
    rx = 0.5
    cx = Complex(0.2, -0.7)
    X = rand(3, 3)
    CX = Complex.(randn(3,3), randn(3,3))

    myisapprox(a, b; kwargs...) = isapprox(a, b; kwargs...)

    noninferrable_inv(x) = x!=rand(size(x)...) ? inv(x) : ""
    ChangesOfVariables.with_logabsdet_jacobian(::typeof(noninferrable_inv), x) = noninferrable_inv(x), with_logabsdet_jacobian(inv, x)[2]
    @test_throws ErrorException @inferred with_logabsdet_jacobian(noninferrable_inv, rand(2, 2))

    test_with_logabsdet_jacobian(inv, rx, ForwardDiff.derivative, atol = 10^-6)
    test_with_logabsdet_jacobian(inv, cx, ForwardDiff.jacobian, rv_and_back, atol = 10^-6)
    test_with_logabsdet_jacobian(inv, X, ForwardDiff.jacobian, atol = 10^-6)
    test_with_logabsdet_jacobian(inv, CX, ForwardDiff.jacobian, rv_and_back, atol = 10^-6)
    test_with_logabsdet_jacobian(inv, CX, ForwardDiff.jacobian, rv_and_back, atol = 10^-6)
    test_with_logabsdet_jacobian(inv, CX, ForwardDiff.jacobian, rv_and_back, compare = myisapprox, atol = 10^-6)
    test_with_logabsdet_jacobian(noninferrable_inv, CX, ForwardDiff.jacobian, rv_and_back, atol = 10^-6)
end
