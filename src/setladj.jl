# This file is a part of ChangesOfVariables.jl, licensed under the MIT License (MIT).


"""
    struct FunctionWithLADJ{F,LADJF} <: Function

A function with an separate function to compute it's `logabddet(J)`.

Do not construct directly, use [`setladj(f, ladjf)`](@ref) instead.
"""
struct FunctionWithLADJ{F,LADJF} <: Function
    f::F
    ladjf::LADJF
end
FunctionWithLADJ(::Type{F}, ladjf::LADJF) where {F,LADJF} = FunctionWithLADJ{Type{F},LADJF}(F,ladjf)
FunctionWithLADJ(f::F, ::Type{LADJF}) where {F,LADJF} = FunctionWithLADJ{F,Type{LADJF}}(f,LADJF)
FunctionWithLADJ(::Type{F}, ::Type{LADJF}) where {F,LADJF} = FunctionWithLADJ{Type{F},Type{LADJF}}(F,LADJF)

(f::FunctionWithLADJ)(x) = f.f(x)

with_logabsdet_jacobian(f::FunctionWithLADJ, x) = f.f(x), f.ladjf(x)


"""
    setladj(f, ladjf)::Function

Return a function that behaves like `f` in general and which has
`with_logabsdet_jacobian(f, x) = f(x), ladjf(x)`.

Useful in cases where [`with_logabsdet_jacobian`](@ref) is not defined
for `f`, or if `f` needs to be assigned a LADJ-calculation that is
only valid within a given context, e.g. only for a
limited argument type/range that is guaranteed by the use case but
not in general, or that is optimized to a custom use case.

For example, `CUDA.CuArray` has no `with_logabsdet_jacobian` defined,
but may be used to switch computing device for a part of a
heterogenous computing function chain. Likewise, one may want to
switch numerical precision for a part of a calculation.

The function (wrapper) returned by `setladj` supports
[`InverseFunctions.inverse`](https://github.com/JuliaMath/InverseFunctions.jl)
if `f` does so.

Example:

```jldoctest setladj
VERSION < v"1.6" || begin # Support for ∘ requires Julia >= v1.6
    # Increases precition before calculation exp:
    foo = exp ∘ setladj(setinverse(Float64, Float32), _ -> 0)

    # A log-value from some low-precision (e.g. GPU) computation:
    log_x = Float32(100)

    # f(log_x) would return Inf32 without going to Float64:
    y, ladj = with_logabsdet_jacobian(foo, log_x) 

    r_log_x, ladj_inv = with_logabsdet_jacobian(inverse(foo), y)

    ladj ≈ 100 ≈ -ladj_inv && r_log_x ≈ log_x
end
# output

true
```
"""
setladj(f, ladjf) = FunctionWithLADJ(_unwrap_f(f), ladjf)
export setladj

_unwrap_f(f) = f
_unwrap_f(f::FunctionWithLADJ) = f.f
