# ChangesOfVariables.jl

```@docs
ChangesOfVariables
```

This package defines the function [`with_logabsdet_jacobian`](@ref). `(y, ladj) = with_logabsdet_jacobian(f, x)` computes both the transformed value of `x` under the transformation `f` and the logarithm of the [volume element](https://en.wikipedia.org/wiki/Volume_element).

`with_logabsdet_jacobian` supports mapped/broadcasted functions (via `Base.Broadcast.BroadcastFunction` or `Base.Fix1`) and function composition.

Implementations of `with_logabsdet_jacobian(f)` for `identity`, `inv`, `adjoint` and `transpose` as well as for `exp`, `log`, `exp2`, `log2`, `exp10`, `log10`, `expm1` and `log1p` are included.
