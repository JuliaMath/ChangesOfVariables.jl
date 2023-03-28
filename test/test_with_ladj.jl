# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).

using ChangesOfVariables
using Test

using LinearAlgebra

using ChangesOfVariables
using ChangesOfVariables: test_with_logabsdet_jacobian

include("getjacobian.jl")


@testset "with_logabsdet_jacobian" begin
    @static if VERSION >= v"1.6"
        _bc_func(f) = Base.Broadcast.BroadcastFunction(f)
    else
        _bc_func(f) = Base.Fix1(broadcast, f)
    end

    @test with_logabsdet_jacobian(sum, rand(5)) == NoLogAbsDetJacobian{typeof(sum),Vector{Float64}}()
    @test with_logabsdet_jacobian(sum ∘ log, 5.0f0) == NoLogAbsDetJacobian{typeof(sum ∘ log),Float32}()
    @test with_logabsdet_jacobian(log ∘ sum, 5.0f0) == NoLogAbsDetJacobian{typeof(log ∘ sum),Float32}()
    @test_throws MethodError _, _ = with_logabsdet_jacobian(sum, rand(5))

    function ChangesOfVariables.with_logabsdet_jacobian(::typeof(foo), x)
        y = foo(x)
        ladj = -x + 2 * log(y)
        (y, ladj)
    end

    x = 4.2
    X = rand(10)
    A = rand(5, 5)
    CA = Complex.(rand(5, 5), rand(5, 5))

    isaprx(a, b; kwargs...) = isapprox(a,b; kwargs...)
    isaprx(a::NTuple{N,Any}, b::NTuple{N,Any}; kwargs...) where N = all(map((a,b) -> isaprx(a, b; kwargs...), a, b))


    test_with_logabsdet_jacobian(foo, x, getjacobian)

    @static if VERSION >= v"1.6"
        test_with_logabsdet_jacobian(log ∘ foo, x, getjacobian)
    end

    @testset "with_logabsdet_jacobian on mapped and broadcasted" begin
        for f in (_bc_func(foo), Base.Fix1(map, foo), Base.Fix1(broadcast, foo))
            for arg in (x, fill(x,), Ref(x), (x,), X)
                test_with_logabsdet_jacobian(f, arg, getjacobian, compare = isaprx)
            end
        end
    end

    @testset "with_logabsdet_jacobian on identity, adjoint and transpose" begin
        for f in (identity, adjoint, transpose)
            for arg in (x, A)
                test_with_logabsdet_jacobian(f, arg, getjacobian)
            end
        end
    end

    @testset "with_logabsdet_jacobian on inv" begin
        for arg in (x, A, CA)
            test_with_logabsdet_jacobian(inv, arg, getjacobian)
        end
    end

    @testset "with_logabsdet_jacobian on log and exp functions" begin
        for f in (+, -, exp, log, exp2, log2, exp10, log10, expm1, log1p)
            test_with_logabsdet_jacobian(f, x, getjacobian)
        end
    end
end
