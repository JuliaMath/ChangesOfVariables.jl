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
    x = Complex(0.2, -0.7)
    y, ladj_y = ChangesOfVariables._auto_with_logabsdet_jacobian(inv, x, ForwardDiff.jacobian, rv_and_back)
    @test y == inv(x)
    @test ladj_y ≈ -4 * log(abs(x))

    X = Complex.(randn(3,3), randn(3,3))
    Y, ladj_Y = ChangesOfVariables._auto_with_logabsdet_jacobian(inv, X, ForwardDiff.jacobian, rv_and_back)
    @test Y == inv(X)
    @test ladj_Y ≈ -4 * 3 * logabsdet(X)[1]

    myisapprox(a, b; kwargs...) = isapprox(a, b; kwargs...)
    test_with_logabsdet_jacobian(inv, 0.5, ForwardDiff.derivative, test_inferred = true, atol = 10^-6)
    test_with_logabsdet_jacobian(inv, rand(2,2), ForwardDiff.jacobian, test_inferred = true, atol = 10^-6)
    test_with_logabsdet_jacobian(inv, X, ForwardDiff.jacobian, rv_and_back, test_inferred = true, atol = 10^-6)
    test_with_logabsdet_jacobian(inv, X, ForwardDiff.jacobian, rv_and_back, test_inferred = false, atol = 10^-6)
    test_with_logabsdet_jacobian(inv, X, ForwardDiff.jacobian, rv_and_back, compare = myisapprox, atol = 10^-6)
end
