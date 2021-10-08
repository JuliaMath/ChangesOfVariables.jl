# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).

"""
    with_logabsdet_jacobian(f, x)

Computes both the transformed value of `x` under the transformation `f` and
the logarithm of the [volume element](https://en.wikipedia.org/wiki/Volume_element).

For `(y, ladj) = with_logabsdet_jacobian(f, x)`, the following must hold true:

* `y == f(x)`
* `ladj` is the `log(abs(det(jacobian(f, x))))`

`with_logabsdet_jacobian` comes with support for broadcasted/mapped functions
(via `Base.Fix1`) and (Julia >=v1.6 only) `ComposedFunction`.

Example:

```julia
foo(x) = inv(exp(-x) + 1)

function ChangesOfVariables.with_logabsdet_jacobian(::typeof(foo), x)
    y = foo(x)
    ladj = -x + 2 * log(y)
    (y, ladj)
end

x = 4.2
y, ladj_y = with_logabsdet_jacobian(foo, x)

X = rand(10)
broadcasted_foo = Base.Fix1(broadcast, foo)
Y, ladj_Y = with_logabsdet_jacobian(broadcasted_foo, X)

# Requires Julia >= v1.6:
z, ladj_z = with_logabsdet_jacobian(log ∘ foo, x)
z == log(foo(x))
ladj_z == ladj_y + with_logabsdet_jacobian(log, y)[2]
```
"""
function with_logabsdet_jacobian end
export with_logabsdet_jacobian

@static if VERSION >= v"1.6"
    function with_logabsdet_jacobian(f::Base.ComposedFunction, x)
        y_inner, ladj_inner = with_logabsdet_jacobian(f.inner, x)
        y, ladj_outer = with_logabsdet_jacobian(f.outer, y_inner)
        (y, ladj_inner + ladj_outer)
    end
end


@inline _get_y(y_with_ladj::NTuple{2,Any,}) = y_with_ladj[1]
@inline _get_ladj(y_with_ladj::NTuple{2,Any}) = y_with_ladj[2]

_with_ladj_on_mapped(map_or_bc::Function, y_with_ladj::Tuple{Any,Real}) = y_with_ladj

function _with_ladj_on_mapped(map_or_bc::Function, y_with_ladj)
    y = map_or_bc(_get_y, y_with_ladj)
    ladj = sum(map_or_bc(_get_ladj, y_with_ladj))
    (y, ladj)
end

function with_logabsdet_jacobian(mapped_f::Base.Fix1{<:Union{typeof(map),typeof(broadcast)}}, X)
    map_or_bc = mapped_f.f
    f = mapped_f.x
    y_with_ladj = map_or_bc(Base.Fix1(with_logabsdet_jacobian, f), X)
    _with_ladj_on_mapped(map_or_bc, y_with_ladj)
end


with_logabsdet_jacobian(::typeof(identity), x) = (identity(x), zero(eltype(x)))

_ndof(::Type{<:Real}) = 1
_ndof(::Type{<:Complex}) = 2
with_logabsdet_jacobian(::typeof(inv), x::Number) = (inv(x), -2 * _ndof(typeof(x)) * log(abs(x)))
with_logabsdet_jacobian(::typeof(inv), A::AbstractMatrix) = (inv(A), -2 * _ndof(eltype(A)) * size(A, 1) * logabsdet(A)[1])

with_logabsdet_jacobian(::typeof(adjoint), x) = (adjoint(x), zero(eltype(x)))
with_logabsdet_jacobian(::typeof(transpose), x) = (transpose(x), zero(eltype(x)))


with_logabsdet_jacobian(::typeof(exp), x) = (exp(x), x)
with_logabsdet_jacobian(::typeof(exp2), x) = (exp2(x), log(2) * x + log(log(oftype(x, 2))))
with_logabsdet_jacobian(::typeof(exp10), x) = (exp10(x), log(10) * x + log(log(oftype(x, 10))))
with_logabsdet_jacobian(::typeof(expm1), x) = (expm1(x), x)

with_logabsdet_jacobian(::typeof(log), x) = (y = log(x); (y, -y))
with_logabsdet_jacobian(::typeof(log2), x) = (y = log2(x); (y, -log(2) * y - log(log(oftype(x, 2)))))
with_logabsdet_jacobian(::typeof(log10), x) = (y = log10(x); (y, -log(10) * y - log(log(oftype(x, 10)))))
with_logabsdet_jacobian(::typeof(log1p), x) = (y = log1p(x); (y, -y))
