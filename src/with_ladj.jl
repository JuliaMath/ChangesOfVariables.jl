# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).

"""
    with_logabsdet_jacobian(f, x)

Computes both the transformed value of `x` under the transformation `f` and
the logarithm of the [volume element](https://en.wikipedia.org/wiki/Volume_element).

For `(y, ladj) = with_logabsdet_jacobian(f, x)`, the following must hold true:

* `y == f(x)`
* `ladj` is the `log(abs(det(jacobian(f, x))))`

`with_logabsdet_jacobian` comes with support for broadcasted/mapped functions
(via `Base.Broadcast.BroadcastFunction` or `Base.Fix1`) and `ComposedFunction`.

If no volume element is defined/applicable, `with_logabsdet_jacobian(f::F, x::T)`
returns [`NoLogAbsDetJacobian{F,T}()`](@ref).

# Examples

```jldoctest a
using ChangesOfVariables

foo(x) = inv(exp(-x) + 1)

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(foo), x)
    y = foo(x)
    ladj = -x + 2 * log(y)
    (y, ladj)
end

x = 4.2
y, ladj_y = with_logabsdet_jacobian(foo, x)

using LinearAlgebra, ForwardDiff
y == foo(x) && ladj_y ≈ log(abs(ForwardDiff.derivative(foo, x)))

# output

true
```

```jldoctest a
X = rand(10)
broadcasted_foo = Base.Broadcast.BroadcastFunction(foo)
Y, ladj_Y = with_logabsdet_jacobian(broadcasted_foo, X)
Y == broadcasted_foo(X) && ladj_Y ≈ logabsdet(ForwardDiff.jacobian(broadcasted_foo, X))[1]

# output

true
```

```jldoctest a
z, ladj_z = with_logabsdet_jacobian(log ∘ foo, x)
z == log(foo(x)) && ladj_z == ladj_y + with_logabsdet_jacobian(log, y)[2]

# output

true
```

Implementations of with_logabsdet_jacobian can be tested (as a
`Test.@testset`) using
[`ChangesOfVariables.test_with_logabsdet_jacobian`](@ref).
"""
function with_logabsdet_jacobian end
export with_logabsdet_jacobian



"""
    struct NoLogAbsDetJacobian{F,T}

An instance `NoLogAbsDetJacobian{F,T}()` signifies that `with_logabsdet_jacobian(::F, ::T)` is not defined.

Constructors:
```julia
NoLogAbsDetJacobian(f, x)
NoLogAbsDetJacobian{F,T}()
```
"""
struct NoLogAbsDetJacobian{F,T} end
export NoLogAbsDetJacobian

@inline NoLogAbsDetJacobian(::F, ::T) where {F,T} = NoLogAbsDetJacobian{F,T}()
@inline NoLogAbsDetJacobian(::Type{F}, ::T) where {F,T} = NoLogAbsDetJacobian{Type{F},T}()
@inline NoLogAbsDetJacobian(::F, ::Type{T}) where {F,T} = NoLogAbsDetJacobian{F,Type{T}}()
@inline NoLogAbsDetJacobian(::Type{F}, ::Type{T}) where {F,T} = NoLogAbsDetJacobian{Type{F},Type{T}}()

with_logabsdet_jacobian(f, x) = NoLogAbsDetJacobian(f, x)


function with_logabsdet_jacobian(f::Base.ComposedFunction, x)
    y_ladj_inner = with_logabsdet_jacobian(f.inner, x)
    if y_ladj_inner isa NoLogAbsDetJacobian
        NoLogAbsDetJacobian(f, x)
    else
        y_inner, ladj_inner = y_ladj_inner
        y_ladj_outer = with_logabsdet_jacobian(f.outer, y_inner)
        if y_ladj_outer isa NoLogAbsDetJacobian
            NoLogAbsDetJacobian(f, x)
        else
            y, ladj_outer = y_ladj_outer
            (y, ladj_inner + ladj_outer)
        end
    end
end


_get_all_first(x) = map(first, x)
_get_second(x) = x[2]
# Use _get_second instead of last, using last causes horrible performance in Zygote here:
_sum_over_second(x) = sum(_get_second, x)

_combine_y_ladj(l, r) = ((l[1]..., r[1]), l[2] + r[2])

_with_ladj_on_mapped(y_with_ladj::Tuple{Any,Any}, ::NoLogAbsDetJacobian) = y_with_ladj

function _with_ladj_on_mapped(y_with_ladj::Tuple{Tuple{Any,Any}, Vararg{Tuple{Any,Any}}}, ::NoLogAbsDetJacobian)
    a, bs = first(y_with_ladj), Base.tail(y_with_ladj)
    return foldl(_combine_y_ladj, bs, init = ((a[1],), a[2]))
end

_with_ladj_on_mapped(y_with_ladj::Tuple{Vararg{Union{Tuple{Any,Any},NoLogAbsDetJacobian}}}, noladj::NoLogAbsDetJacobian) = noladj

function _with_ladj_on_mapped(y_with_ladj::AbstractArray{<:Tuple{Any,Any}}, ::NoLogAbsDetJacobian)
    y = _get_all_first(y_with_ladj)
    ladj = _sum_over_second(y_with_ladj)
    (y, ladj)
end

_with_ladj_on_mapped(::Any, noladj::NoLogAbsDetJacobian) = noladj

function with_logabsdet_jacobian(mapped_f::Base.Broadcast.BroadcastFunction, X)
    f = mapped_f.f
    y_with_ladj = broadcast(Base.Fix1(with_logabsdet_jacobian, f), X)
    _with_ladj_on_mapped(y_with_ladj, NoLogAbsDetJacobian(mapped_f, X))
end

function with_logabsdet_jacobian(mapped_f::Base.Fix1{<:Union{typeof(map),typeof(broadcast)}}, X)
    map_or_bc = mapped_f.f
    f = mapped_f.x
    y_with_ladj = map_or_bc(Base.Fix1(with_logabsdet_jacobian, f), X)
    _with_ladj_on_mapped(y_with_ladj, NoLogAbsDetJacobian(mapped_f, X))
end


with_logabsdet_jacobian(::typeof(identity), x) = (identity(x), zero(eltype(x)))

_ndof(::Type{<:Real}) = 1
_ndof(::Type{<:Complex}) = 2
with_logabsdet_jacobian(::typeof(inv), x::Number) = (inv(x), -2 * _ndof(typeof(x)) * log(abs(x)))
with_logabsdet_jacobian(::typeof(inv), A::AbstractMatrix) = (inv(A), -2 * _ndof(eltype(A)) * size(A, 1) * logabsdet(A)[1])

with_logabsdet_jacobian(::typeof(adjoint), x) = (adjoint(x), zero(eltype(x)))
with_logabsdet_jacobian(::typeof(transpose), x) = (transpose(x), zero(eltype(x)))

with_logabsdet_jacobian(::typeof(+), x) = (+(x), zero(eltype(x)))
with_logabsdet_jacobian(::typeof(-), x) = (-(x), zero(eltype(x)))

with_logabsdet_jacobian(::typeof(exp), x) = (exp(x), x)
with_logabsdet_jacobian(::typeof(exp2), x) = (exp2(x), log(2) * x + log(log(oftype(x, 2))))
with_logabsdet_jacobian(::typeof(exp10), x) = (exp10(x), log(10) * x + log(log(oftype(x, 10))))
with_logabsdet_jacobian(::typeof(expm1), x) = (expm1(x), x)

with_logabsdet_jacobian(::typeof(log), x) = (y = log(x); (y, -y))
with_logabsdet_jacobian(::typeof(log2), x) = (y = log2(x); (y, -log(2) * y - log(log(oftype(x, 2)))))
with_logabsdet_jacobian(::typeof(log10), x) = (y = log10(x); (y, -log(10) * y - log(log(oftype(x, 10)))))
with_logabsdet_jacobian(::typeof(log1p), x) = (y = log1p(x); (y, -y))
